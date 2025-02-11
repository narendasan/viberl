import argparse
import os
from pathlib import Path

import jax
from rejax import PPO

import wandb
from rl_sandbox.utils import (build_eval_callback, create_checkpointer,
                              create_eval_logger, create_wandb_logger, argparser,
                              generate_experiment_config)

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)

wandb.init(
    project="rl-sandbox",
    name=config["experiment"]["experiment_name"],
    tags=config["experiment"]["tags"],
    config=config
)

root_key = jax.random.PRNGKey(config["experiment"]["root_seed"])
agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])


ppo = PPO.create(**config["algorithm"])
eval_callbacks = build_eval_callback(ppo, [
    create_eval_logger(),
    create_wandb_logger(),
    create_checkpointer(config["experiment"]["ckpt_dir"], config["experiment"]["experiment_name"])
])

ppo = ppo.replace(eval_callback=eval_callbacks)

train_agents = jax.jit(jax.vmap(ppo.train))#.lower(agent_keys).compile()

print("Training...")
train_state, evaluation = train_agents(agent_keys)
print(train_state, evaluation)
