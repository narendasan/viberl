import copy
import logging
import json
from jax._src.core import check_eqn
from jax._src.source_info_util import current
from optax._src.wrappers import chex
from orbax.checkpoint._src.checkpointers.async_checkpointer import checkpoint
import tomli_w
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import numpy as np
from flax import struct
from flax.serialization import to_state_dict
from rejax.algos import Algorithm
import orbax.checkpoint as ocp

from rejax.evaluate import EvalState
import wandb
from rl_sandbox.utils._checkpoints import generate_checkpointer_options
from rl_sandbox.utils._readable_hash import generate_phrase_hash
from rl_sandbox.utils.types import EvalCallback, PolicyEvalResult

"""
Creating an evaluation callback. Two examples are available:

1. The Wandb Logger which logs training metrics to Wandb.
usage:
```python
wandb_logger = create_wandb_logger()
```

2. The Checkpointer which saves model checkpoints and evaluation results.
usage:
```python
ckpt = create_checkpointer("path/to/checkpoints", "experiment_name")
```

To combine multiple callbacks, pass them as a list to `build_eval_callback`:
```python
callback = build_eval_callback(algorithm, [wandb_logger, ckpt])
```

To create your own callback:
1. Define a function that takes (algorithm, train_state, rng, eval_results) as input
2. Process the evaluation results as needed
3. Return an empty tuple () if no training data needs to be collected
4. Use jax.experimental.io_callback for any I/O operations

The callback will be called during training to evaluate policy performance.
"""

_LOGGER = logging.getLogger(__name__)

def build_eval_callback(algo_instance: Algorithm, fs: List[EvalCallback]) -> Callable[[Algorithm, struct.PyTreeNode, jax.Array], Tuple]:
    """Build an evaluation callback from a list of callback functions.

    This function combines multiple evaluation callbacks into a single callback function.
    During training, each callback in the list will be called sequentially to process
    evaluation results.

    Args:
        algo_instance: Algorithm instance that will be evaluated
        fs: List of callback functions to execute during evaluation

    Returns:
        Callable: Combined callback function that executes all provided callbacks and takes:
            - algorithm: The training algorithm instance
            - train_state: Current training state
            - key: Random number generator key
    """
    policy_eval_callback = algo_instance.eval_callback
    def eval_callback(algo: Algorithm, train_state: struct.PyTreeNode, key: jax.Array) -> Tuple[Tuple[Any, ...]]:
        policy_result = PolicyEvalResult(*policy_eval_callback(algo, train_state, key))
        results = []
        for f in fs:
            results.append(f(algo, train_state, key, policy_result))
        return tuple(results)

    return eval_callback

def create_eval_logger() -> EvalCallback:
    """Create a callback for logging evaluation metrics to console.

        This function creates a callback that logs policy evaluation metrics to the console.
        During training, it logs the current step, total steps, mean episode length and mean
        return at each evaluation step.

        Returns:
            EvalCallback: Callback function for console logging that takes:
                - algorithm: The training algorithm instance
                - train_state: Current training state
                - key: Random number generator key
                - eval_results: Results from policy evaluation
    """
    def eval_logger(a: Algorithm, train_state: struct.PyTreeNode, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:

        def log(current_step: jax.Array, total_steps: int, mean_return: float, mean_length: float, id: jax.Array) -> None:
            _LOGGER.info(f"[{current_step.item()}/{total_steps}](id: {generate_phrase_hash(id[1])}): mean return: {mean_return} mean length: {mean_length}")
            print(f"[{current_step.item()}/{total_steps}](id: {generate_phrase_hash(id[1])}): mean return: {mean_return} mean length: {mean_length}")

        jax.experimental.io_callback(
            log,
            (),
            train_state.global_step,
            a.total_timesteps,
            eval_results.returns.mean(),
            eval_results.lengths.mean(),
            train_state.seed
        )
        return ()
    return eval_logger


def create_wandb_logger() -> EvalCallback:
    """Create a callback for logging to Weights & Biases.

    This function creates a callback that logs policy evaluation metrics to Weights & Biases (wandb).
    During training, it logs mean episode length and mean return at each evaluation step.

    Returns:
        EvalCallback: Callback function for logging to wandb that takes:
            - algorithm: The training algorithm instance
            - train_state: Current training state
            - key: Random number generator key
            - eval_results: Results from policy evaluation
    """
    def wandb_logger(a: Algorithm, train_state: struct.PyTreeNode, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def log(step: int, data: Dict[str, float]) -> None:
            # io_callback returns np.array, which wandb does not like.
            # In jax 0.4.27, this becomes a jax array, should check when upgrading...
            step = step.item()
            wandb.log(data, step=step)

        jax.experimental.io_callback(
            log,
            (),  # result_shape_dtypes (wandb.log returns None)
            train_state.global_step,
            {"mean_episode_length": eval_results.lengths.mean(), "mean_return": eval_results.returns.mean()},
        )

        # Since we log to wandb, we don't want to return anything that is collected
        # throughout training
        return ()
    return wandb_logger

def create_checkpointer(ckpt_dir: str | Path, exp_name: str | Path, max_to_keep: int=50) -> EvalCallback:
    """Create a callback for saving model checkpoints.

    This function creates a checkpointer callback that saves model checkpoints and evaluation
    results during training. For each evaluation, it saves the current model state and metrics
    to a checkpoint file. It also maintains a "best" checkpoint based on mean returns.

    Args:
        ckpt_dir: Directory path where checkpoints will be saved
        exp_name: Name of the experiment for organizing checkpoints

    Returns:
        EvalCallback: Callback function for saving checkpoints during evaluation that takes:
            - algorithm: The training algorithm instance
            - train_state: Current training state
            - key: Random number generator key
            - eval_results: Results from policy evaluation
    """
    exp_prefix = Path(exp_name)
    ckpt_dir_prefix = Path(ckpt_dir)
    if not str(ckpt_dir_prefix).startswith("/"):
        ckpt_dir_prefix = Path(os.getcwd()) / ckpt_dir_prefix
    exp_path = ckpt_dir_prefix / exp_prefix
    exp_path.mkdir(parents=True, exist_ok=True)

    def checkpointer(algo: Algorithm, train_state: struct.PyTreeNode, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def create_checkpoint(current_step: int, t: struct.PyTreeNode, e: PolicyEvalResult, id: jax.Array, total_timesteps: int) -> None:
            # TODO: Move this into rejax
            options = generate_checkpointer_options(max_to_keep=max_to_keep)
            with ocp.CheckpointManager(
                exp_path / generate_phrase_hash(id[1]),
                options=options,
            ) as ocp_checkpointer:
                ocp_checkpointer.save(
                    current_step,
                    args=ocp.args.StandardSave(t),
                    metrics={
                        "mean_returns": e.returns.mean().item(),
                        "mean_lengths": e.lengths.mean().item()
                    }
                )
            return

        jax.experimental.io_callback(
            create_checkpoint,
            (),
            train_state.global_step,
            train_state,
            eval_results,
            copy.deepcopy(train_state.seed),
            algo.total_timesteps,
        )

        return ()
    return checkpointer

def create_checkpointer_from_config(config: Dict[str, Any]) -> EvalCallback:
    """Create a callback for saving model checkpoints from configuration.

    This function creates a checkpointer callback and saves the experiment configuration
    to the checkpoint directory.

    Args:
        config: Dictionary containing experiment configuration, expected to have:
            - experiment.ckpt_dir: Directory to save checkpoints
            - experiment.experiment_name: Name of the experiment

    Returns:
        EvalCallback: Callback function for saving checkpoints during evaluation
    """
    callback = create_checkpointer(ckpt_dir=config["experiment"]["ckpt_dir"], exp_name=config["experiment"]["experiment_name"], max_to_keep=config["experiment"]["max_ckpt_to_keep"])

    ckpt_dir_prefix = Path(config["experiment"]["ckpt_dir"])
    exp_prefix = Path(config["experiment"]["experiment_name"])

    with open(ckpt_dir_prefix / f"{exp_prefix}" / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    return callback
