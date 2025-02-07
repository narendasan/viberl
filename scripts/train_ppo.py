import jax

import wandb
from rejax import PPO
from rl_sandbox.utils import create_wandb_logger

CONFIG = {
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

wandb.init(project="my-awesome-project", config=CONFIG)

ppo = PPO.create(**CONFIG)
ppo = ppo.replace(eval_callback=create_wandb_logger(ppo))

rng = jax.random.PRNGKey(0)
print("Compiling...")
compiled_train = jax.jit(ppo.train).lower(rng).compile()
print("Training...")
compiled_train(rng)
