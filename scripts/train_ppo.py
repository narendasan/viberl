import logging
from functools import partial

import jax
import jax.numpy as jnp
from rejax import PPO

import wandb
from rl_sandbox.env._trajectories import collect_trajectories
from rl_sandbox.env._visualize import collect_rollouts
from rl_sandbox.utils import (argparser, build_eval_callback,
                              create_checkpointer_from_config, create_eval_logger,
                              create_wandb_logger, generate_experiment_config)

logging.basicConfig(level=logging.INFO)

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)

wandb.init(
    project="rl-sandbox",
    group=config["experiment"]["experiment_name"],
    tags=config["experiment"]["tags"],
    config=config
)
# JAX handles reproducablity through the key system. Starting from a root key (can be thought of as a seed)
# keys can be split to control PRNG across a vector of agents
# Here we create N splits of the root key, one for each agent we will train
# Under the hood, each PPO instance will also split their key M times for each of the envs it will train across
root_key = jax.random.key(config["experiment"]["root_seed"])
agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])

# Here we create a vector of N agents that we will train seeded with their own key derived from the root key
algo = PPO.create(**config["algorithm"])
# We then insert the callbacks for logging and reporting on training process into each agent
# These transforms are functional so you get a new agent out instead of modifying in place
algo = algo.replace(eval_callback=build_eval_callback(algo, [
    create_eval_logger(),
    create_wandb_logger(),
    create_checkpointer_from_config(config)
]))

print("Training...")
# We then can vectorize across NxM instances of agents and envs and train these in parallel
vmap_train = jax.jit(jax.vmap(algo.train))
train_states, results = vmap_train(agent_keys)
print(results)
#policies = jax.vmap(lambda a, ts: PPO.make_act(a, ts))(agents, train_states)

vmap_collect_trajectories = partial(collect_trajectories,
    algo=algo,
    env_name=config["algorithm"]["env"],
    env_config=config["algorithm"]["env_params"],
    num_envs=config["algorithm"]["num_envs"],
    max_steps_in_episode=1000
)

t_results, trajectories = jax.vmap(vmap_collect_trajectories)(train_states, agent_keys)
print(t_results)
#vis = Visualizer(env, env_params, state_seq, cum_rewards)
#is.animate("anim.gif")
print(trajectories)
breakpoint()
