import logging
import os

import flax
import jax
import jax.numpy as jnp

import viberl

#from rejax import PPO
from viberl.algorithms._vppo import VPPO
from viberl.env import render_gymnax
from viberl.utils import (
    argparser,
    generate_experiment_config,
    setup_logger,
)

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

# Use PRNGKey since theres some limitations with .key
root_key = jax.random.PRNGKey(config["experiment"]["root_seed"])
agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])

algo = VPPO.create(**config["algorithm"])

jit_train = jax.jit(jax.vmap(algo.train))
train_states, results = jit_train(agent_keys)

agents = viberl.utils.tree_unstack(train_states)

# Extract and flatten Actor params
params = agents[0].actor_ts.params
flattened_params, restore_fn = jax.flatten_util.ravel_pytree(params)
print(flattened_params.shape)

reconstructed_params = restore_fn(flattened_params)


def make_act(
    ppo: VPPO,
    ts: flax.training.train_state.TrainState,
    actor_params: flax.core.FrozenDict,
):
    @jax.jit
    def act(obs, rng):
        if getattr(ppo, "normalize_observations", False):
            obs = ppo.normalize_obs(ts.obs_rms_state, obs)

        obs = jnp.expand_dims(obs, 0)
        action = ppo.actor.apply(actor_params, obs, rng, method="act")
        return jnp.squeeze(action)

    return act


# reconstructed_train_state = train_state.replace(actor_ts=train_state.actor_ts.replace(params=reconstructed_params))
# restored_algo = PPO.create(**config["algorithm"])
# jit_policy = jax.jit(restored_algo.make_act(reconstructed_train_state))

jit_policy = make_act(algo, agents[0], reconstructed_params)

vis, reward = render_gymnax(jit_policy, config, root_key)
vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/restored_ppo.gif")
