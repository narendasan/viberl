from typing import Optional
import os
import json
import logging
from pathlib import Path
import pickle as pkl
import jax
from jax._src.core import Value
from jax._src.lax import lax
import numpy as np
from flax import struct
from orbax.checkpoint.transform_utils import RestoreArgs
from rejax.algos import Algorithm
from wandb import run
from rl_sandbox.utils._readable_hash import generate_phrase_hash
import orbax.checkpoint as ocp

_LOGGER = logging.getLogger(__name__)

def generate_checkpointer_options(max_to_keep: int=50) -> ocp.checkpoint_manager.CheckpointManagerOptions:
    return ocp.checkpoint_manager.CheckpointManagerOptions(
        best_fn = lambda x: x["mean_returns"],
        best_mode = "max",
        max_to_keep=max_to_keep
    )


def load_ckpt(algo: Algorithm, ckpt_dir: str, experiment_name: str, key: Optional[jax.Array]=None, run_name: Optional[str]=None, tag: str | int="best") -> struct.PyTreeNode:
    """Load a model checkpoint from disk.

    Args:
        ckpt_dir (str): Checkpoint directory path
        experiment_name (str): Name of the experiment
        key (jax.Array): JAX random key array
        tag (Union[str, int], optional): Tag identifier for the checkpoint. Defaults to "best", also accepts "latest" or a integer which represents the step.

    Returns:
        struct.PyTreeNode: Loaded model checkpoint state
    """
    ts = algo.init_state(key)
    options = generate_checkpointer_options()

    if key is not None and run_name is not None:
        raise ValueError("Both key and name cannot be provided")
    elif key is not None:
        id = jax.random.key_data(key)
        phrase_hash = generate_phrase_hash(id[1])
    elif run_name is not None:
        # TODO: Use the run_name but still generate an acceptable TrainState
        phrase_hash = run_name
    else:
        raise ValueError("Either key or run_name must be provided")

    if not ckpt_dir.startswith("/"):
        ckpt_dir_path = Path(os.getcwd()) / ckpt_dir
    else:
        ckpt_dir_path = Path(ckpt_dir)

    with jax.default_device(jax.devices('gpu')[0]):
        with ocp.CheckpointManager(ckpt_dir_path / experiment_name / phrase_hash, options=options) as ocp_checkpointer:
            if tag == "best":
                train_state = ocp_checkpointer.restore(ocp_checkpointer.best_step(), args=ocp.args.StandardRestore(ts))
            elif tag == "latest":
                train_state = ocp_checkpointer.restore(ocp_checkpointer.latest_step(), args=ocp.args.StandardRestore(ts))
            elif isinstance(tag, int):
                train_state = ocp_checkpointer.restore(tag, args=ocp.args.StandardRestore(ts))
            elif isinstance(tag, str) and tag.isdigit():
                train_state = ocp_checkpointer.restore(int(tag), args=ocp.args.StandardRestore(ts))
            else:
                raise ValueError(f"Invalid tag: {tag}, must be 'best', 'latest', an integer, or a string representing an integer")

        _LOGGER.info(f"Loaded checkpoint {tag} for {phrase_hash}: {train_state}")

    print(train_state)
    return train_state
