# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_dp_lang import DPLangConfig
from .modeling_dp_lang import DPLangPolicy
from .processor_dp_lang import make_dp_lang_pre_post_processors

__all__ = [
    "DPLangConfig",
    "DPLangPolicy",
    "make_dp_lang_pre_post_processors",
]