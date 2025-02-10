import pickle as pkl
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from flax.serialization import to_state_dict
import jax
from jax.tree_util import PyTreeDef
from rejax.algos import Algorithm

import wandb
from rl_sandbox.utils.types import EvalCallback, PolicyEvalResult


def build_eval_callback(algo_instance: Algorithm, fs: List[EvalCallback]) -> Callable[[Algorithm, PyTreeDef, jax.Array], Tuple]:
    policy_eval_callback = algo_instance.eval_callback
    def eval_callback(algo: Algorithm, train_state: PyTreeDef, key: jax.Array) -> Tuple[Tuple[Any, ...]]:
        policy_result = PolicyEvalResult(*policy_eval_callback(algo, train_state, key))
        results = []
        for f in fs:
            results.append(f(algo, train_state, key, policy_result))
        return tuple(results)

    return eval_callback

def create_wandb_logger() -> EvalCallback:
    def wandb_logger(a: Algorithm, train_state: PyTreeDef, rng: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def log(step: int, data: Dict[str, float]):
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

    def checkpointer(algo: Algorithm, train_state: PyTreeDef, rng: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def create_checkpoint(train_state: PyTreeDef):
            with open(exp_path / f"{train_state.global_step}_ckpt.pkl", "wb") as f:
                train_state_dict = to_state_dict(train_state)
                pkl.dump(train_state_dict, f)

            with open(exp_path / f"{train_state.global_step}_results.txt", "w") as f:
                f.write(str(eval_results))
            return

        jax.experimental.io_callback(
            create_checkpoint,
            (),
            train_state,
        )

        return ()
    return checkpointer
