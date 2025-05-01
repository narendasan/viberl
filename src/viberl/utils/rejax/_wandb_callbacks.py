import copy
from typing import Any, Dict, Tuple

import jax
import wandb
from flax import struct
from rejax.algos import Algorithm

from viberl.utils._readable_hash import generate_phrase_hash
from viberl.utils.types import PolicyEvalResult
from viberl.utils._eval_callbacks import EvalCallback


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
        a: Algorithm,
        train_state: struct.PyTreeNode,
        key: jax.Array,
        eval_results: PolicyEvalResult,
    ) -> Tuple:
        def log(id: jax.Array, step: jax.Array, data: Dict[str, float]) -> None:
            wandb.init(  # type: ignore
                project="rl-sandbox",
                group=config["experiment"]["experiment_name"],
                tags=config["experiment"]["tags"],
                config=config,
                resume="allow",
                reinit=True,
                id=f"{generate_phrase_hash(id[1])}-{config['experiment']['experiment_name']}",
            )
            # io_callback returns np.array, which wandb does not like.
            # In jax 0.4.27, this becomes a jax array, should check when upgrading...
            step = step.item()
            wandb.log(data, step=step)  # type: ignore
            wandb.finish()  # type: ignore

        jax.experimental.io_callback(
            log,
            (),  # result_shape_dtypes (wandb.log returns None)
            copy.deepcopy(train_state.seed),
            train_state.global_step,
            {
                "mean_episode_length": eval_results.lengths.mean(),
                "mean_return": eval_results.returns.mean(),
            },
        )

        # Since we log to wandb, we don't want to return anything that is collected
        # throughout training
        return ()

    return wandb_logger
