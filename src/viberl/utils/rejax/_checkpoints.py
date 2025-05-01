import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import orbax.checkpoint as ocp
import tomli_w
from flax import struct
from rejax.algos import Algorithm

from viberl.utils._readable_hash import generate_phrase_hash
from viberl.utils.types import PolicyEvalResult
from viberl.utils._eval_callbacks import EvalCallback

_LOGGER = logging.getLogger(__name__)


def generate_checkpointer_options(
    max_to_keep: int = 50,
) -> ocp.checkpoint_manager.CheckpointManagerOptions:
    return ocp.checkpoint_manager.CheckpointManagerOptions(
        best_fn=lambda x: x["mean_returns"],
        best_mode="max",
        max_to_keep=max_to_keep,
        enable_async_checkpointing=True,
    )


def load_ckpt(
    algo: Algorithm,
    ckpt_dir: str,
    experiment_name: str,
    key: Optional[jax.Array] = None,
    run_name: Optional[str] = None,
    step: str | int = "best",
    rng: Optional[jax.Array] = None,
) -> struct.PyTreeNode:
    """Load a model checkpoint from disk.

    Args:
        ckpt_dir (str): Checkpoint directory path
        experiment_name (str): Name of the experiment
        key (jax.Array): JAX random key array
        step (Union[str, int], optional): Tag identifier for the checkpoint. Defaults to "best", also accepts "latest" or a integer which represents the step.

    Returns:
        struct.PyTreeNode: Loaded model checkpoint state
    """
    options = generate_checkpointer_options()
    if key is not None:
        ts = algo.init_state(key)
    elif rng is not None:
        rng, _key = jax.random.split(rng)
        ts = algo.init_state(jax.random.key_data(_key))
    elif rng is None and key is None:
        raise ValueError("Either key or rng must be provided")

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

    with jax.default_device(jax.devices("gpu")[0]):
        with ocp.CheckpointManager(
            ckpt_dir_path / experiment_name / phrase_hash, options=options
        ) as ocp_checkpointer:
            if step == "best":
                train_state = ocp_checkpointer.restore(
                    ocp_checkpointer.best_step(), args=ocp.args.StandardRestore(ts)
                )
            elif step == "latest":
                train_state = ocp_checkpointer.restore(
                    ocp_checkpointer.latest_step(), args=ocp.args.StandardRestore(ts)
                )
            elif isinstance(step, int):
                train_state = ocp_checkpointer.restore(
                    step, args=ocp.args.StandardRestore(ts)
                )
            elif isinstance(step, str) and step.isdigit():
                train_state = ocp_checkpointer.restore(
                    int(step), args=ocp.args.StandardRestore(ts)
                )
            else:
                raise ValueError(
                    f"Invalid step: {step}, must be 'best', 'latest', an integer, or a string representing an integer"
                )

        _LOGGER.info(f"Loaded checkpoint {step} for {phrase_hash}: {train_state}")

    return train_state


def create_checkpointer(
    ckpt_dir: str | Path, exp_name: str | Path, max_to_keep: int = 50
) -> EvalCallback:
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

    def checkpointer(
        algo: Algorithm,
        train_state: struct.PyTreeNode,
        key: jax.Array,
        eval_results: PolicyEvalResult,
    ) -> Tuple:
        def create_checkpoint(
            current_step: int,
            t: struct.PyTreeNode,
            e: PolicyEvalResult,
            id: jax.Array,
            total_timesteps: int,
        ) -> None:
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
                        "mean_lengths": e.lengths.mean().item(),
                    },
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
    callback = create_checkpointer(
        ckpt_dir=config["experiment"]["ckpt_dir"],
        exp_name=config["experiment"]["experiment_name"],
        max_to_keep=config["experiment"]["max_ckpt_to_keep"],
    )

    ckpt_dir_prefix = Path(config["experiment"]["ckpt_dir"])
    exp_prefix = Path(config["experiment"]["experiment_name"])

    with open(ckpt_dir_prefix / f"{exp_prefix}" / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    return callback
