from viberl.utils import types
from viberl.utils._checkpoints import (
    create_checkpointer,
    create_checkpointer_from_config,
    load_ckpt,
)
from viberl.utils._eval_callbacks import (
    build_eval_callback,
    create_eval_logger,
)
from viberl.utils._exp_manager import argparser, argparser_for_eval, generate_experiment_config
from viberl.utils._logger import setup_logger

from viberl.utils._pytrees import tree_stack, tree_unstack

__all__ = [
    "build_eval_callback",
    "create_checkpointer",
    "create_checkpointer_from_config",
    "create_eval_logger",
    "types",
    "generate_experiment_config",
    "argparser",
    "argparser_for_eval",
    "load_ckpt",
    "setup_logger",
    "tree_stack",
    "tree_unstack"
]


import importlib
import warnings

if importlib.util.find_spec("wandb"):
    from viberl.utils._wandb_callbacks import create_wandb_logger
    __all__ += ["create_wandb_logger"]
else:
    warnings.warn("Install wandb to use the wandb logging system")


if importlib.util.find_spec("mlflow"):
    from viberl.utils._mlflow_callbacks import create_mlflow_logger
    __all__ += ["create_mlflow_logger"]
else:
    warnings.warn("Install mlflow to use the mlflow logging system")
