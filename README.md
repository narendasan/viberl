# VibeRL

VibeRL is a toolkit for reinforcement learning, designed to facilitate the use of standalone RL implementations such as CleanRL and ReJAX in experiments primarily in JAX.
It provides common utilities for checkpointing, logging, and visualization of RL policies focusing on vmaped training loops allowing researchers to concurrently evaluate multiple experiment configurations side by side.

## Installation

To install VibeRL, run the following command:

```bash
uv pip install -e .[mlflow]
```

## Usage

```pythoon
import logging
import os

import jax
from brax import envs
from brax.io import html
from orbax.checkpoint import tree
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
from viberl.utils import tree_unstack
from viberl.env import (render_brax, render_gymnax)

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

root_key = jax.random.key(config["experiment"]["root_seed"])
agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])

algo = PPO.create(**config["algorithm"])
# We insert the callbacks for logging and reporting on training process into each agent
# These transforms are functional so you get a new agent out instead of modifying in place
algo = algo.replace(eval_callback=build_eval_callback(algo, [
    create_eval_logger(),
    create_mlflow_logger(config),
    create_checkpointer_from_config(config)
]))

# We then can vectorize across NxM instances of agents and envs and train these in parallel
# This can just be run as JIT, but further gains can be gotten from lowering and AOT compiling the training function
_LOGGER.info("Compiling training function...")
vmap_train = jax.jit(jax.vmap(algo.train)).lower(agent_keys).compile()

_LOGGER.info("Training...")
train_states, results = vmap_train(agent_keys)
print(results)

# Separate the vectorized training state into individual training states for each agetn
agent_ts = tree_unstack(train_states)

# Visualize the policy for agent 0
policy = algo.make_act(agent_ts[0])
jit_policy = jax.jit(policy)
vis, reward = render_gymnax(jit_policy, config, root_key)
vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/train_ppo.gif")
```
