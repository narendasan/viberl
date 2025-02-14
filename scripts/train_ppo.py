from functools import partial

import jax
from rejax import PPO

import wandb
from rl_sandbox.env import collect_trajectories
from rl_sandbox.utils import (argparser, build_eval_callback,
                              create_checkpointer, create_eval_logger,
                              create_wandb_logger, generate_experiment_config)

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
agents = jax.vmap(lambda _: PPO.create(**config["algorithm"]))(agent_keys)
# We then insert the callbacks for logging and reporting on training process into each agent
# These transforms are functional so you get a new agent out instead of modifying in place
agents = jax.vmap(lambda a: PPO.replace(a, eval_callback=build_eval_callback(a, [
    create_eval_logger(),
    create_wandb_logger(),
    create_checkpointer(config["experiment"]["ckpt_dir"], config["experiment"]["experiment_name"])
])))(agents)

print("Training...")
# We then can vectorize across NxM instances of agents and envs and train these in parallel
train_states, results = jax.jit(jax.vmap(PPO.train))(agents, agent_keys)

#policies = jax.vmap(lambda a, ts: PPO.make_act(a, ts))(agents, train_states)

vmap_collect_trajectories = partial(collect_trajectories, env_name=config["algorithm"]["env"], env_config=config["algorithm"]["env_params"])
trajectories = jax.vmap(vmap_collect_trajectories)(agents, train_states, agent_keys)

#vis = Visualizer(env, env_params, state_seq, cum_rewards)
#is.animate("anim.gif")
