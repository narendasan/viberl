import logging
import os

from gymnax.visualize import Visualizer
import gymnax
import jax
from jax import numpy as jnp
from mlflow.protos.service_pb2 import Run
from numpy.core.numeric import roll
from rejax import PPO

from rl_sandbox.utils import (
    argparser_for_eval,
    build_eval_callback,
    create_checkpointer_from_config,
    create_eval_logger,
    create_mlflow_logger,
    generate_experiment_config,
    load_ckpt,
    setup_logger,
)

parser = argparser_for_eval()
args = parser.parse_args()

config = generate_experiment_config(args.config_file)
root_key = jax.random.key(config["experiment"]["root_seed"])

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

env, env_params = gymnax.make(config["algorithm"]["env"])

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

algo0 = PPO.create(**config["algorithm"])

train_state = load_ckpt(
    algo0,
    config["experiment"]["ckpt_dir"],
    args.experiment,
    run_name=args.seed_name,
    step=args.step,
    rng=root_key)

root_key, reset_key = jax.random.split(root_key)

inference_fn = algo0.make_act(train_state)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
obs, state = jit_env_reset(reset_key, env_params)
cum_reward = 0
done = False
info = None

while True:
    rollout.append(state)
    rng, step_rng, act_rng = jax.random.split(root_key, 3)
    act = jit_inference_fn(obs, act_rng)
    obs, state, reward, done, info = jit_env_step(step_rng, state, act, env_params)
    cum_reward += reward
    if done:
        break


os.makedirs(f"{os.getcwd()}/{config['experiment']['results_dir']}", exist_ok=True)
vis = Visualizer(env, env_params, rollout, jnp.array([cum_reward]))
vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/{args.seed_name}_{args.step}_{args.experiment}.gif")
_LOGGER.info(f"Saved [{args.seed_name} (step: {args.step})]: Reward: {cum_reward}")
