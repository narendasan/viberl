from viberl.utils.rejax._checkpoints import (
    create_checkpointer,
    create_checkpointer_from_config,
    load_ckpt,
)
from viberl.utils.rejax._eval_callbacks import (
    build_eval_callback,
    create_eval_logger,
)

__all__ = [
    "build_eval_callback",
    "create_checkpointer",
    "create_checkpointer_from_config",
    "create_eval_logger",
    "load_ckpt",
]

import importlib
import warnings

if importlib.util.find_spec("wandb"):  # type: ignore[attr-defined]
    from viberl.utils.rejax._wandb_callbacks import create_wandb_logger

    __all__ += ["create_wandb_logger"]
else:
    warnings.warn("Install wandb to use the wandb logging system")


if importlib.util.find_spec("mlflow"):  # type: ignore[attr-defined]
    from viberl.utils.rejax._mlflow_callbacks import create_mlflow_logger

    __all__ += ["create_mlflow_logger"]
else:
    warnings.warn("Install mlflow to use the mlflow logging system")
