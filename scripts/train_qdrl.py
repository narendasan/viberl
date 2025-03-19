import logging
import os

import jax
from brax import envs
from brax.io import html
from orbax.checkpoint import tree
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
from rl_sandbox.utils import tree_unstack
from rl_sandbox.env import (render_brax, render_gymnax)

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

#Use PRNGKey since theres some limitations with .key
root_key = jax.random.PRNGKey(config["experiment"]["root_seed"])

algo = PPO.create(**config["algorithm"])

jit_train = jax.jit(algo.train)
train_state, results = jit_train(root_key)

flattened_train_state, restore_fn = jax.flatten_util.ravel_pytree(train_state)
print(flattened_train_state.shape)

reconstructed_train_state = restore_fn(flattened_train_state)

restored_algo = PPO.create(**config["algorithm"])
jit_policy = jax.jit(restored_algo.make_act(reconstructed_train_state))

vis, reward = render_gymnax(jit_policy, config, root_key)
vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/restored_ppo.gif")
