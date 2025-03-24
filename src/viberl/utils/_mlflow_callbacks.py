import copy
from typing import Any, Dict, Tuple

import jax
import mlflow
from flax import struct
from rejax.algos import Algorithm

from viberl.utils._readable_hash import generate_phrase_hash
from viberl.utils.types import EvalCallback, PolicyEvalResult


def create_mlflow_logger(config: Dict[str, Any]) -> EvalCallback:
    """Create a callback for logging to MLFlow.

    This function creates a callback that logs policy evaluation metrics to MLFlow.
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
    mlflow_runs = {}
    experiment_id = mlflow.create_experiment(config["experiment"]["experiment_name"])
    parent_run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=config['experiment']['experiment_name'],
        #tags=config["experiment"]["tags"],
    )

    def mlflow_logger(a: Algorithm, train_state: struct.PyTreeNode, key: jax.Array, eval_results: PolicyEvalResult) -> Tuple:
        def log(id: jax.Array, step: int, data: Dict[str, float]) -> None:
            shard_id = generate_phrase_hash(id[1])
            if shard_id not in mlflow_runs:
                with parent_run:
                    run = mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=f"{shard_id}-{config['experiment']['experiment_name']}",
                        #tags=config["experiment"]["tags"],
                        nested=True,
                        parent_run_id=parent_run.info.run_id
                    )
                    mlflow_runs[shard_id] = run.info.run_id
                    mlflow.log_params(config, run_id=mlflow_runs[shard_id])

            mlflow.log_metrics(data, step=step.item(), run_id=mlflow_runs[shard_id])


        jax.experimental.io_callback(
            log,
            (),  # result_shape_dtypes (wandb.log returns None)
            copy.deepcopy(train_state.seed),
            train_state.global_step,
            {"mean_episode_length": eval_results.lengths.mean(), "mean_return": eval_results.returns.mean()},
        )

        # Since we log to wandb, we don't want to return anything that is collected
        # throughout training
        return ()
    return mlflow_logger
