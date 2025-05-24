import copy
from typing import Any, Dict, Tuple

import logging

import jax
from flax import struct

import wandb
from viberl.utils._readable_hash import generate_phrase_hash
from viberl.utils.types import EvalCallback, PolicyEvalResult

_LOGGER = logging.getLogger(__name__)

_WANDB_INSTANCES = {}

def create_wandb_logger(config: Dict[str, Any]) -> EvalCallback:
    """Create a callback for logging to Weights & Biases.

    This function creates a callback that logs policy evaluation metrics to Weights & Biases (wandb).
    During training, it logs mean episode length and mean return at each evaluation step.

    NOTE: This function is very slow as it requires sequential logging of data to wandb.

    Returns:
        EvalCallback: Callback function for logging to wandb that takes:
            - algorithm: The training algorithm instance
            - train_state: Current training state
            - key: Random number generator key
            - eval_results: Results from policy evaluation
    """
    config = copy.deepcopy(config)


    def wandb_logger(
        state: struct.PyTreeNode,
        cfg: struct.PyTreeNode,
        eval_results: PolicyEvalResult,
        rollout: struct.PyTreeNode,
    ) -> Tuple:
        def log(id: jax.Array, step: jax.Array, data: Dict[str, float]) -> None:
            phrase_id = generate_phrase_hash(id[1])
            if phrase_id not in _WANDB_INSTANCES:
                run = wandb.init(  # type: ignore
                    project="viberl",
                    group=config["experiment"]["experiment_name"],
                    tags=config["experiment"]["tags"],
                    config=config,
                    resume="allow",
                    reinit="create_new",
                    id=f"{phrase_id}_{config['experiment']['experiment_name']}",
                )
                _WANDB_INSTANCES[phrase_id] = run
            else:
                run = _WANDB_INSTANCES[phrase_id]
            # io_callback returns np.array, which wandb does not like.
            # In jax 0.4.27, this becomes a jax array, should check when upgrading...
            step = step.item()
            run.log(data, step=step)  # type: ignore

        train_metrics = state.train_metrics.compute()
        eval_metrics = state.eval_metrics.compute()
        rollout_metrics = state.rollout_metrics.compute()

        metrics = {
                f"training/{key}": value for key, value in train_metrics.items()
            } | {
                f"eval/{key}": value for key, value in eval_metrics.items()
            } | {
                f"rollout/{key}": value for key, value in rollout_metrics.items()
            }

        _LOGGER.debug(metrics)

        jax.experimental.io_callback(
            log,
            (),  # result_shape_dtypes (wandb.log returns None)
            copy.deepcopy(state.actor_critic.actor.id),
            state.global_step,
            metrics,
        )

        # Since we log to wandb, we don't want to return anything that is collected
        # throughout training
        return ()

    return wandb_logger
