import json
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import numpy as np
from flax import struct
from flax.serialization import to_state_dict
from rejax.algos import Algorithm

import wandb
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

def build_eval_callback(algo_instance: Algorithm, fs: List[EvalCallback]) -> Callable[[Algorithm, struct.PyTreeNode, jax.Array], Tuple]:
    policy_eval_callback = algo_instance.eval_callback
    def eval_callback(algo: Algorithm, train_state: struct.PyTreeNode, key: jax.Array) -> Tuple[Tuple[Any, ...]]:
        policy_result = PolicyEvalResult(*policy_eval_callback(algo, train_state, key))
        results = []
        for f in fs:
            results.append(f(algo, train_state, key, policy_result))
        return tuple(results)

    return eval_callback

def create_eval_logger() -> EvalCallback:
    def eval_logger(a: Algorithm, train_state: struct.PyTreeNode, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:

        def log(current_step: jax.Array, total_steps: int, mean_return: float, mean_length: float, id: jax.Array) -> None:
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

def create_checkpointer(ckpt_dir: str | Path, exp_name: str | Path) -> EvalCallback:
    exp_prefix = Path(exp_name)
    ckpt_dir_prefix = Path(ckpt_dir)
    exp_path = ckpt_dir_prefix / exp_prefix
    exp_path.mkdir(parents=True, exist_ok=True)
    def checkpointer(algo: Algorithm, train_state: struct.PyTreeNode, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def create_checkpoint(t: struct.PyTreeNode, e: PolicyEvalResult, id: jax.Array, total_timesteps: int, mean_returns: float) -> None:
            gs = str(t.global_step)
            ts = str(total_timesteps)
            zs = len(ts) - len(gs)
            gs = "0"*zs + gs
            mean_returns = e.returns.mean()
            mean_lengths = e.lengths.mean()

            def write_checkpoint(id: jax.Array, t: struct.PyTreeNode, mean_returns: float, mean_lengths: float, tag: str):
                (exp_path / f"{generate_phrase_hash(id[1])}").mkdir(parents=True, exist_ok=True)
                with open(exp_path / f"{generate_phrase_hash(id[1])}" / f"{tag}_ckpt.pkl", "wb") as f:
                    train_state_dict = to_state_dict(t)
                    pkl.dump(train_state_dict, f)

                with open(exp_path / f"{generate_phrase_hash(id[1])}"/ f"{tag}_results.json", "w") as f:
                    json.dump({
                        "mean_returns": mean_returns.item(),
                        "mean length": mean_lengths.item(),
                        "timestep": int(t.global_step.item()),
                        "seed": np.asarray(id).tolist(),
                    }, f)

            write_checkpoint(id, t, mean_returns, mean_lengths, gs)

            if not os.path.isfile(exp_path / f"{generate_phrase_hash(id[1])}" / "best_results.json"):
                write_checkpoint(id, t, mean_returns, mean_lengths, "best")

            else:
                with open(exp_path / f"{generate_phrase_hash(id[1])}" / "best_results.json", "r") as f:
                    best = json.load(f)

                if best["mean_returns"] < mean_returns:
                    write_checkpoint(id, t, mean_returns, mean_lengths, "best")

            return

        jax.experimental.io_callback(
            create_checkpoint,
            (),
            train_state,
            eval_results,
            train_state.seed,
            algo.total_timesteps,
            eval_results.returns.mean(),
        )

        return ()
    return checkpointer
