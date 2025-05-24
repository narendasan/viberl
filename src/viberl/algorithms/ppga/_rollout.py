from typing import Dict, Tuple

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


def flatten_vec_rollout(orig: Rollout, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Rollout:
    return Rollout(
        obs=orig.obs.reshape((-1,) + obs_shape),
        actions=orig.actions.reshape((-1,) + action_shape),
        logprobs=orig.logprobs.reshape((-1, orig.logprobs.shape[-1])),
        rewards=orig.rewards.reshape((-1, orig.rewards.shape[-1])),
        dones=orig.dones.reshape((-1, orig.dones.shape[-1])),
        truncated=orig.truncated.reshape((-1, orig.truncated.shape[-1])),
        values=orig.values.reshape((-1, orig.values.shape[-1])),
    )


def make_empty_rollout(rollout_len: int, num_envs: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> Rollout:
    return Rollout(
        obs=jnp.zeros((rollout_len, num_envs) + obs_shape),
        actions=jnp.zeros((rollout_len, num_envs) + action_shape),
        logprobs=jnp.zeros((rollout_len, num_envs, 1)),
        rewards=jnp.zeros((rollout_len, num_envs, 1)),
        dones=jnp.zeros((rollout_len, num_envs, 1)),
        truncated=jnp.zeros((rollout_len, num_envs, 1)),
        values=jnp.zeros((rollout_len, num_envs, 1)),
    )
