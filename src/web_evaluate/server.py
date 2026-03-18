import logging
import socket
from pathlib import Path
import torch
import numpy as np
import asyncio
import websockets
import traceback
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from typing import Any

from lerobot.policies.factory import make_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser

from contextlib import nullcontext
from termcolor import colored

import http
from dataclasses import asdict, dataclass, field

from lerobot.policies.factory import (
    get_policy_class,
    make_pre_post_processors,
    make_policy,
)
from lerobot.utils.import_utils import register_third_party_plugins

import websockets
import websockets.asyncio.server as _server


from lerobot.policies.pretrained import PreTrainedPolicy


from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
import functools

import msgpack

# import debugpy
# try:
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


from logging import getLogger

logger = getLogger(__name__)


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    # --- ADDED CODE BELOW ---
    if isinstance(obj, Path):
        return str(obj)
    # ------------------------

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


@dataclass
class WebPolicyServerConfig:
    """Configuration for WebPolicyServer.

    This class defines all configurable parameters for the WebPolicyServer,
    including networking settings and action chunking specifications.
    """

    # 首先定义没有默认值的字段
    dataset: DatasetConfig

    # 然后定义有默认值的字段
    host: str = field(
        default="127.0.0.1", metadata={"help": "Host address to bind the server to"}
    )
    port: int = field(
        default=10123, metadata={"help": "Port number to bind the server to"}
    )
    seed: int = field(default=42, metadata={"help": "Seed"})
    policy: PreTrainedConfig | None = None

    def __post_init__(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)
        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class WebPolicyServer:
    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        device: str,
        host: str = "127.0.0.1",
        port: int = 10123,
    ):
        """初始化策略服务器"""
        self.policy = policy
        self.policy.reset()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        self.host = host
        self.port = port
        logging.info(
            f"PolicyServer initialized with device={device}, host={host}, port={port}"
        )
        logging.info(f"Policy type: {type(policy).__name__}")

        # 记录模型输入输出特征
        logging.info(f"Policy input features: {policy.config.input_features}")
        logging.info(f"Policy output features: {policy.config.output_features}")

    def server_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(asdict(self.policy.config)))

        while True:
            try:
                message = await websocket.recv()
                if isinstance(message,bytes) and message==b'reset':
                    logging.info(f"Policy Reset")
                    self.policy.reset()
                    await websocket.send(b'reset_complete')
                    continue

                obs = unpackb(message)

                processed_obs = self._prepare_observation(obs)

                # The pre-processing pipeline prepares the input data for the model by:
                # 1. Renaming features.
                # 2. Normalizing the input and output features based on dataset statistics.
                # 3. Adding a batch dimension.
                # 4. Moving the data to the specified device.
                processed_obs = self.preprocessor(processed_obs)
                single_action_tensor = self.policy.select_action(processed_obs)
                # The post-processing pipeline handles the model's output by:
                # 1. Moving the data to the CPU.
                # 2. Unnormalizing the output features to their original scale.
                processed_action_dict = self._prepare_action(single_action_tensor)

                postprocessed_action_dict = self.postprocessor(processed_action_dict)

                postprocessed_action = postprocessed_action_dict[ACTION]
                
                send_processed_action = packb(postprocessed_action.squeeze().numpy(), use_bin_type=True)
                await websocket.send(send_processed_action)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _prepare_observation(self, obs: dict) -> dict:
        processed_obs = {}
        #   vlabench 和 libero客户端发过来的obs在key name和value format上都是一致的，可以统一处理
        
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # 转换为torch tensor并放到设备上
                tensor_value = torch.from_numpy(np.array(value, copy=True)).to(self.device)
                # 处理图像数据（根据客户端代码，图像已经是[C, H, W]格式， 数值[0,1]）
                if ("observation.image" in key and tensor_value.ndim == 3) or \
                    ("observation.state" in key and tensor_value.ndim == 1) or \
                    ("trinsic" in key and tensor_value.ndim == 1):
                    # 添加批次维度
                    tensor_value = tensor_value.unsqueeze(0)
                    processed_obs[key] = tensor_value
            elif key == "task" and isinstance(value, str):
                processed_obs["task"] = [value] 
            else:
                processed_obs[key] = value
        
        return processed_obs

    def _prepare_action(self, action_tensor: torch.Tensor) -> dict:
        action = {}
        if isinstance(action_tensor, torch.Tensor):
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)
            action[ACTION]=action_tensor
        else:
            raise ValueError("action_tensor must be torch.Tensor")
        
        return action
    
def _health_check(
        connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        # Continue with the normal request handling.
        return None


@parser.wrap()
def server_main(cfg: WebPolicyServerConfig):
    """主函数"""
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info("Making policy.")
    # print(f"=====dataset root: {cfg.dataset.root} =====")
    logging.info(f"Dataset root: {cfg.dataset.root}")
    dataset_metadata = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root
    )
    
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset_metadata)
    policy.eval()

    logging.info("Making preprocessor & postprocessor")
    # logging.info(f"Pretrained path: {cfg.policy.pretrained_path}")
    preprocessor = PolicyProcessorPipeline.from_pretrained(cfg.policy.pretrained_path,config_filename="policy_preprocessor.json")
    postprocessor = PolicyProcessorPipeline.from_pretrained(cfg.policy.pretrained_path,config_filename="policy_postprocessor.json")
  
    ctx = (
        torch.autocast(device_type=device.type if device != "cpu" else "cpu")
        if cfg.policy.use_amp
        else nullcontext()
    )

    with torch.no_grad(), ctx:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")
        server = WebPolicyServer(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            host=cfg.host,
            port=cfg.port,
        )

        try:
            server.server_forever()

        except KeyboardInterrupt:
            logging.info("Server stopped by user")
        except Exception as e:
            logging.error(f"Server error: {e}")
            import traceback

            logging.error(traceback.format_exc())


if __name__ == "__main__":
    # # 调试
    # import debugpy
    # debugpy.listen(9599)
    # print("Waiting for debugger attach on port 9599...")
    # debugpy.wait_for_client()
    
    register_third_party_plugins()
    init_logging()
    server_main()
