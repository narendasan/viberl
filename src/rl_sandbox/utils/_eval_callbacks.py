import pickle as pkl
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
from flax.serialization import to_state_dict
from jax.tree_util import PyTreeDef
from rejax.algos import Algorithm

import wandb
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

def build_eval_callback(algo_instance: Algorithm, fs: List[EvalCallback]) -> Callable[[Algorithm, PyTreeDef, jax.Array], Tuple]:
    policy_eval_callback = algo_instance.eval_callback
    def eval_callback(algo: Algorithm, train_state: PyTreeDef, key: jax.Array) -> Tuple[Tuple[Any, ...]]:
        policy_result = PolicyEvalResult(*policy_eval_callback(algo, train_state, key))
        results = []
        for f in fs:
            results.append(f(algo, train_state, key, policy_result))
        return tuple(results)

    return eval_callback

def create_eval_logger() -> EvalCallback:
    def eval_logger(a: Algorithm, train_state: PyTreeDef, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:

        def log(current_step: int, total_steps: int, mean_return: float, mean_length: float, id: int) -> None:
            print(f"[{current_step.item()}/{total_steps}](id: {id}): mean return: {mean_return} mean length: {mean_length}")

        jax.experimental.io_callback(
            log,
            (),
            train_state.global_step,
            a.total_timesteps,
            eval_results.returns.mean(),
            eval_results.lengths.mean(),
            key[0]
        )
        return ()
    return eval_logger


def create_wandb_logger() -> EvalCallback:
    def wandb_logger(a: Algorithm, train_state: PyTreeDef, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
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

def create_checkpointer(ckpt_dir: str | Path, exp_name: str | Path) -> EvalCallback:
    exp_prefix = Path(exp_name)
    ckpt_dir_prefix = Path(ckpt_dir)
    exp_path = ckpt_dir_prefix / exp_prefix
    exp_path.mkdir(parents=True, exist_ok=True)

    def checkpointer(algo: Algorithm, train_state: PyTreeDef, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def create_checkpoint(t: PyTreeDef, e: PolicyEvalResult, id: int) -> None:
            with open(exp_path / f"{t.global_step}_id{id}_ckpt.pkl", "wb") as f:
                train_state_dict = to_state_dict(t)
                pkl.dump(train_state_dict, f)

            with open(exp_path / f"{t.global_step}_id{id}_results.txt", "w") as f:
                f.write(f"mean returns: {e.returns.mean()}\n")
                f.write(f"mean length: {e.lengths.mean()}\n")
            return

        jax.experimental.io_callback(
            create_checkpoint,
            (),
            train_state,
            eval_results,
            key[0]
        )

        return ()
    return checkpointer
