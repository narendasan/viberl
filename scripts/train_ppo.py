import logging
import os

import jax
from brax import envs
from brax.io import html
from rejax import PPO

from rl_sandbox.utils import (
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

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

# wandb.init(
#     project="rl-sandbox",
#     group=config["experiment"]["experiment_name"],
#     tags=config["experiment"]["tags"],
#     config=config
# )
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
    create_mlflow_logger(config),
    create_checkpointer_from_config(config)
]))

_LOGGER.info("Training...")
# We then can vectorize across NxM instances of agents and envs and train these in parallel
# This can just be run as JIT, but further gains can be gotten from lowering and AOT compiling the training function
vmap_train = jax.jit(jax.vmap(algo.train)).lower(agent_keys).compile()
train_states, results = vmap_train(agent_keys)
print(results)

env = envs.create(env_name="ant", backend="positional")

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

algo0 = PPO.create(**config["algorithm"])
train_state = load_ckpt(algo0, config["experiment"]["ckpt_dir"], config["experiment"]["experiment_name"], key=train_states.seed[0], tag="best")
inference_fn = algo0.make_act(train_state)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
reward = 0
for _ in range(10000):
    reward += state.reward
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

os.makedirs(f"{os.getcwd()}/{config['experiment']['results_dir']}", exist_ok=True)
html.save(f"{os.getcwd()}/{config['experiment']['results_dir']}/{config['experiment']['experiment_name']}.html", env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
_LOGGER.info(f"Saved: {reward}")
