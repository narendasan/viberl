import logging
import os

import jax
from brax import envs
from brax.io import html
from rejax import PPO

from viberl.utils import (
    argparser,
    build_eval_callback,
    create_checkpointer_from_config,
    create_eval_logger,
    create_mlflow_logger,
    generate_experiment_config,
    load_ckpt,
    setup_logger,
)

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)
root_key = jax.random.key(config["experiment"]["root_seed"])

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

env = envs.create(env_name="walker2d", backend="positional")

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

algo0 = PPO.create(**config["algorithm"])
train_state = load_ckpt(
    algo0,
    config["experiment"]["ckpt_dir"],
    "test_experiment1-20250222200407-seed42-steps10000000-lr0.0003-test",
    run_name="serene_rhythm_transforms_pessimistically",
    tag="best",
    rng=root_key)
inference_fn = algo0.make_act(train_state)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
state = jit_env_reset(rng=root_key)
reward = 0
for _ in range(10000):
    reward += state.reward
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(root_key)
    act = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

os.makedirs(f"{os.getcwd()}/{config['experiment']['results_dir']}", exist_ok=True)
html.save(f"{os.getcwd()}/{config['experiment']['results_dir']}/{config['experiment']['experiment_name']}.html", env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
_LOGGER.info(f"Saved: {reward}")
