import os
from pathlib import Path

import jax
from rejax import PPO

import wandb
from rl_sandbox.utils import (build_eval_callback, create_checkpointer,
                              create_wandb_logger)

CONFIG = {
    "exp_config": {
        "root_seed": 42,
        "num_agent_seeds": 128,
        "ckpt_dir": Path(os.getcwd()) / "ckpts" ,
        "exp_name": "test_exp",
        "tags": ["test"]
    },
    "ppo_config" : {
        "env": "brax/ant",
        "env_params": {"backend": "positional"},
        "agent_kwargs": {"activation": "relu"},
        "total_timesteps": 10_000_000,
        "eval_freq": 100_000,
        "num_envs": 2_000,
        "num_steps": 5,
        "num_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 0.0003,
        "max_grad_norm": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    }
}

wandb.init(
    project="rl-sandbox",
    name=CONFIG["exp_config"]["exp_name"],
    tags=CONFIG["exp_config"]["tags"],
    config=CONFIG
)

root_key = jax.random.PRNGKey(CONFIG["exp_config"]["root_seed"])
agent_keys = jax.random.split(root_key, CONFIG["exp_config"]["num_agent_seeds"])


ppo = PPO.create(**CONFIG["ppo_config"])
eval_callbacks = build_eval_callback(ppo, [
    create_wandb_logger(),
    create_checkpointer(CONFIG["exp_config"]["ckpt_dir"], CONFIG["exp_config"]["exp_name"])
])

ppo = ppo.replace(eval_callback=eval_callbacks)

train_agents = jax.jit(jax.vmap(ppo.train))#.lower(agent_keys).compile()

print("Training...")
train_state, evaluation = train_agents(agent_keys)
print(train_state, evaluation)
