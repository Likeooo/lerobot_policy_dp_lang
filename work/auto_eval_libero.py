import os
import time
import subprocess
import signal
import shutil

os.environ["DEBUG_BLACK_IMAGE"] = "0"  # 设置为 "1" 启用全黑图像调试模式
os.environ["DEBUG_ZERO_STATE"] = "0"  # 设置为 "1" 启用全0状态调试模式

# --- 路径配置 ---
TASK_NAME = "libero_10" # libero_object libero_spatial libero_goal libero_10
MODEL_TYPE = "dp_lang" 
MODEL_NAME = f"libero_multiview_6camera_gpu_8_bz_32"
ROOT_PATH=f"/meta_eon_cfs/home/ljh/.cache/huggingface/lerobot/libero/libero_multiview" ### 模型在那个数据集下训练就用哪个
PORT = "10504"
CAMERA_PERTURBATION = "large" # fix small medium large
SAVE_DIR = f"eval_{CAMERA_PERTURBATION}_cam"

# 项目路径配置 (请确保这些是文件夹的绝对路径)
SERVER_PROJECT_PATH = "/meta_eon_cfs/home/ljh/code/lerobot_policy_dp_lang"
CLIENT_PROJECT_PATH = "/meta_eon_cfs/home/ljh/code/LIBERO-multiview"
CKPT_ROOT = f"{SERVER_PROJECT_PATH}/outputs/{MODEL_TYPE}/{MODEL_NAME}/checkpoints"
LOG_FILE = f"{SERVER_PROJECT_PATH}/outputs/{MODEL_TYPE}/{MODEL_NAME}/{SAVE_DIR}_{TASK_NAME}_evaluated_checkpoints.txt"
POLL_INTERVAL = 600

# --- Conda 环境 Python 路径配置 ---
# 请通过 `conda activate env_name && which python` 获取
SERVER_PYTHON = "/meta_eon_cfs/home/ljh/miniconda3/envs/dp_lang/bin/python"
CLIENT_PYTHON = "/meta_eon_cfs/home/ljh/.conda/envs/libero/bin/python"

# --- 显存与负载配置 ---
MIN_FREE_MEMORY = 4000  # 单位 MiB
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

def get_gpu_status():
    """
    获取 GPU 状态
    返回格式: { gpu_id: {'free_mem': int, 'utilization': int} }
    """
    try:
        # 查询项：index, 剩余显存, GPU利用率
        cmd = "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
        
        gpu_status = {}
        for line in output.split('\n'):
            idx, free, util = map(int, line.split(', '))
            if idx in GPU_LIST:
                gpu_status[idx] = {'free_mem': free, 'utilization': util}
        return gpu_status
    except Exception as e:
        print(f"⚠️ Error querying NVIDIA-SMI: {e}")
        return {}

def find_best_gpu():
    """
    策略：
    1. 显存必须大于 MIN_FREE_MEMORY
    2. 在满足条件的显卡中，选择 utilization (利用率) 最低的
    3. 如果满足条件的显卡的利用率都大于90%, 则选择显存最大的那个，并发出警告
    """
    while True:
        status = get_gpu_status()
        candidates = []
        
        for gpu_id, info in status.items():
            if info['free_mem'] >= MIN_FREE_MEMORY:
                candidates.append((gpu_id, info['utilization']))
        
        if candidates:
            if all(util > 90 for _, util in candidates):
                print("⚠️ All candidate GPUs have high utilization (>90%). Selecting the one with the most free memory.")
                candidates.sort(key=lambda x: status[x[0]]['free_mem'], reverse=True)
                best_gpu = candidates[0][0]
                best_util = candidates[0][1]
                print(f"✅ Best GPU found (high utilization): ID {best_gpu} (Utilization: {best_util}%, Free Mem: {status[best_gpu]['free_mem']}MiB)")
                return best_gpu
            # 按利用率升序排序，取第一个
            candidates.sort(key=lambda x: x[1])
            best_gpu = candidates[0][0]
            best_util = candidates[0][1]
            print(f"✅ Best GPU found: ID {best_gpu} (Utilization: {best_util}%, Free Mem: {status[best_gpu]['free_mem']}MiB)")
            return best_gpu
        
        print(f"❌ No GPU meets the memory requirement ({MIN_FREE_MEMORY} MiB). Waiting 60s...")
        time.sleep(60)

def run_evaluation(ckpt_path, ckpt_name):
    # 自动择优
    gpu_id = find_best_gpu()
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)

    server_bin = os.path.dirname(SERVER_PYTHON)
    client_bin = os.path.dirname(CLIENT_PYTHON)
    env["PATH"] = os.pathsep.join([client_bin, server_bin, env.get("PATH", "")])

    ffmpeg_path = shutil.which("ffmpeg", path=env["PATH"])
    if ffmpeg_path:
        env["FFMPEG_BINARY"] = ffmpeg_path
    else:
        print("⚠️ ffmpeg not found in subprocess PATH. Video saving may fail.")
    # env["MUJOCO_GL"] = "egl"
    # env["PYOPENGL_PLATFORM"] = "egl"
    # env["__EGL_VENDOR_LIBRARY_DIRS"] = os.path.expanduser("~/.local/lib/nvidia-egl-570.86.15/egl_vendor.d")
    # env["LD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib/nvidia-egl-570.86.15:$LD_LIBRARY_PATH")
    # env["no_proxy"] = "localhost,127.0.0.1"

    print(f"🚀 [Eval Start] {ckpt_name} on GPU {gpu_id}")
    
    # 启动 Server
    server_cmd = [
        SERVER_PYTHON, f"{SERVER_PROJECT_PATH}/src/web_evaluate/server.py",
        f"--dataset.repo_id=libero/{TASK_NAME}_multiview",
        f"--dataset.root={ROOT_PATH}",
        f"--policy.path={ckpt_path}/pretrained_model",
        f"--port={PORT}",
    ]
    
    server_proc = subprocess.Popen(server_cmd, env=env, cwd=SERVER_PROJECT_PATH)
    time.sleep(30) # 预留模型加载时间

    print(f"✅ Server started for checkpoint {ckpt_name} on GPU {gpu_id}. Starting client evaluation...")
    # 启动 Client
    client_cmd = [
        CLIENT_PYTHON, f"{CLIENT_PROJECT_PATH}/scripts/eval_lerobot.py",
        "--args.task_suite_name", TASK_NAME,
        "--args.save_dir", SAVE_DIR,
        "--args.port", PORT,
        "--args.camera_perturbation", CAMERA_PERTURBATION
    ]
    
    try:
        # 使用 Popen 启动 Client 以实时查看日志
        client_proc = subprocess.Popen(client_cmd, env=env, cwd=CLIENT_PROJECT_PATH)
        # 等待 Client 进程结束
        client_proc.wait()
        
        # 检查 Client 是否成功退出
        if client_proc.returncode != 0:
            raise subprocess.CalledProcessError(client_proc.returncode, client_cmd)
        # 记录成功
        with open(LOG_FILE, "a") as f:
            f.write(f"{ckpt_name}\n")
        print(f"✨ [Eval Done] {ckpt_name}")
    except subprocess.CalledProcessError:
        print(f"❌ [Eval Failed] {ckpt_name}")
        
    finally:
        # 释放资源
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print(f"♻️  Resources released for GPU {gpu_id}")

def main():
    print("🛰️  VLA Auto-Evaluation Monitor Started.")
    while True:
        if not os.path.exists(CKPT_ROOT):
            time.sleep(60)
            continue
        
        print("Start scanning new checkpoint......")
        # 扫描并排序
        all_ckpts = [d for d in os.listdir(CKPT_ROOT) if os.path.isdir(os.path.join(CKPT_ROOT, d)) and d != "last" ]
        all_ckpts.sort(key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
        
        # 读取已完成列表
        evaluated = set()
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                evaluated = set(line.strip() for line in f)

        for ckpt in all_ckpts:
            if ckpt not in evaluated:
                ckpt_path = os.path.abspath(os.path.join(CKPT_ROOT, ckpt))
                if os.path.exists(os.path.join(ckpt_path, "pretrained_model")):
                    run_evaluation(ckpt_path, ckpt)
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()