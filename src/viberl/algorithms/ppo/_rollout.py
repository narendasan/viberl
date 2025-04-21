from typing import Tuple, Dict

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

    def __len__(self):
        return self.obs.shape[0]

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "obs": self.obs.shape,
            "actions": self.actions.shape,
            "logprobs": self.logprobs.shape,
            "rewards": self.rewards.shape,
            "dones": self.dones.shape,
            "truncated": self.truncated.shape,
            "values": self.values.shape,
        }


def make_empty_rollout(rollout_len: int, num_envs: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Rollout:
    return Rollout(
        obs=jnp.zeros((rollout_len, num_envs) + obs_shape),
        actions=jnp.zeros((rollout_len, num_envs) + action_shape),
        logprobs=jnp.zeros((rollout_len, num_envs)),
        rewards=jnp.zeros((rollout_len, num_envs)),
        dones=jnp.zeros((rollout_len, num_envs)),
        truncated=jnp.zeros((rollout_len, num_envs)),
        values=jnp.zeros((rollout_len, num_envs, 1)),
    )
