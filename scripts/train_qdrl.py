import logging
import os

import jax
from rejax import PPO
import flax
import jax.numpy as jnp

from viberl.utils import (
    argparser,
    generate_experiment_config,
    setup_logger,
)
from viberl.env import (render_gymnax)

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

# Extract and flatten Actor params
params = train_state.actor_ts.params
flattened_params, restore_fn = jax.flatten_util.ravel_pytree(params)
print(flattened_params.shape)

reconstructed_params = restore_fn(flattened_params)


def make_act(ppo: PPO, ts: flax.training.train_state.TrainState, actor_params: flax.core.FrozenDict):
    @jax.jit
    def act(obs, rng):
        if getattr(ppo, "normalize_observations", False):
            obs = ppo.normalize_obs(ts.obs_rms_state, obs)

        obs = jnp.expand_dims(obs, 0)
        action = ppo.actor.apply(actor_params, obs, rng, method="act")
        return jnp.squeeze(action)

    return act


#reconstructed_train_state = train_state.replace(actor_ts=train_state.actor_ts.replace(params=reconstructed_params))
#restored_algo = PPO.create(**config["algorithm"])
#jit_policy = jax.jit(restored_algo.make_act(reconstructed_train_state))

jit_policy = make_act(algo, train_state, reconstructed_params)

vis, reward = render_gymnax(jit_policy, config, root_key)
vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/restored_ppo.gif")
