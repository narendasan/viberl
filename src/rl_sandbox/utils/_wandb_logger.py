import wandb
import jax
from typing import Any, Callable, Tuple
from rejax import Algorithm

def create_wandb_logger(algo) -> Callable[[Algorithm, Any, Any], Tuple]:
    eval_callback = algo.eval_callback
    def wandb_logger(a, train_state, rng):
        lengths, returns = eval_callback(a, train_state, rng)

        def log(step, data):
            # io_callback returns np.array, which wandb does not like.
            # In jax 0.4.27, this becomes a jax array, should check when upgrading...
            step = step.item()
            wandb.log(data, step=step)

        jax.experimental.io_callback(
            log,
            (),  # result_shape_dtypes (wandb.log returns None)
            train_state.global_step,
            {"episode_length": lengths.mean(), "return": returns.mean()},
        )

        # Since we log to wandb, we don't want to return anything that is collected
        # throughout training
        return ()
    return wandb_logger
