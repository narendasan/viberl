import logging
from typing import Any, Callable, List, Tuple

import jax
from flax import struct
from rejax.algos import Algorithm

from viberl.utils._readable_hash import generate_phrase_hash
from viberl.utils.types import PolicyEvalResult

EvalCallback = Callable[
    [Algorithm, struct.PyTreeNode, jax.Array, PolicyEvalResult], Tuple
]

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


def build_eval_callback(
    algo_instance: Algorithm, fs: List[EvalCallback]
) -> Callable[[Algorithm, struct.PyTreeNode, jax.Array], Tuple]:
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

    def eval_callback(
        algo: Algorithm, train_state: struct.PyTreeNode, key: jax.Array
    ) -> Tuple[Tuple[Any, ...], ...]:
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

    def eval_logger(
        a: Algorithm,
        train_state: struct.PyTreeNode,
        key: jax.Array,
        eval_results: PolicyEvalResult,
    ) -> Tuple:

        def log(
            current_step: jax.Array,
            total_steps: int,
            mean_return: float,
            mean_length: float,
            id: jax.Array,
        ) -> None:
            _LOGGER = logging.getLogger(__name__)
            _LOGGER.info(
                f"[{current_step.item()}/{total_steps}](id: {generate_phrase_hash(id[1])}): mean return: {mean_return} mean length: {mean_length}"
            )

        jax.experimental.io_callback(
            log,
            (),
            train_state.global_step,
            a.total_timesteps,
            eval_results.returns.mean(),
            eval_results.lengths.mean(),
            train_state.seed,
        )
        return ()

    return eval_logger
