from typing import Tuple

import chex
import jax.numpy as jnp


@chex.dataclass
class Rollout:
    obs: chex.Array
    actions: chex.Array
    logprobs: chex.Array
    rewards: chex.Array
    dones: chex.Array
    truncated: chex.Array
    values: chex.Array
    measures: chex.Array
    len: int


def make_empty_rollout(rollout_len: int, num_envs: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], num_measures: int) -> Rollout:
    return Rollout(
        obs=jnp.zeros((rollout_len, num_envs) + obs_shape),
        actions=jnp.zeros((rollout_len, num_envs) + action_shape),
        logprobs=jnp.zeros((rollout_len, num_envs)),
        rewards=jnp.zeros((rollout_len, num_envs)),
        dones=jnp.zeros((rollout_len, num_envs)),
        truncated=jnp.zeros((rollout_len, num_envs)),
        values=jnp.zeros((rollout_len, num_envs)),
        measures=jnp.zeros((rollout_len, num_envs, num_measures)),
        len=rollout_len
    )
