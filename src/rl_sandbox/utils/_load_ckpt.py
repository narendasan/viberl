from typing import Optional
import json
import logging
from pathlib import Path
import pickle as pkl
import jax
from jax._src.core import Value
import numpy as np
from flax import struct
from wandb import run
from rl_sandbox.utils._readable_hash import generate_phrase_hash
import orbax.checkpoint as ocp

_LOGGER = logging.getLogger(__name__)

def load_ckpt(ckpt_dir: str, experiment_name: str, key: Optional[jax.Array]=None, run_name: Optional[str]=None,  tag: str="best") -> struct.PyTreeNode:
    """Load a model checkpoint from disk.

    Args:
        ckpt_dir (str): Checkpoint directory path
        experiment_name (str): Name of the experiment
        key (jax.Array): JAX random key array
        tag (str, optional): Tag identifier for the checkpoint. Defaults to "best".

    Returns:
        struct.PyTreeNode: Loaded model checkpoint state
    """
    if key is not None and run_name is not None:
        raise ValueError("Both key and name cannot be provided")
    elif key is not None:
        id = jax.random.key_data(key)
        phrase_hash = generate_phrase_hash(id[1])
    elif run_name is not None:
        phrase_hash = run_name
    else:
        raise ValueError("Either key or run_name must be provided")


    with open(Path(ckpt_dir) / experiment_name / phrase_hash / f"{tag}_ckpt.pkl", "rb") as f:
        train_state = pkl.load(f)
    with open(Path(ckpt_dir) / experiment_name / phrase_hash / f"{tag}_results.json", "rb") as f:
        train_reuslts = json.load(f)

    _LOGGER.info(f"Loaded checkpoint {tag} for {phrase_hash}: {train_state}")

    return train_state
