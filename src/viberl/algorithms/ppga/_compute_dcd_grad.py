from typing import Optional, Tuple, Sequence

import jax
import jax.numpy as jnp

import chex
import gymnax
from jaxlib.mlir.ir import VectorType
import numpy as np
from numpy.lib.function_base import vectorize
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork

from viberl.models._actor import VectorizedActor, ActorMLP
from viberl.models._critic import QDCritic, CriticMLP

from viberl.algorithms.ppga._rollout import Rollout, make_empty_rollout

def calculate_discounted_sum(
    x: jax.Array,
    dones: jax.Array,
    discount: float,
    prev_x: Optional[jax.Array] = None
) -> jax.Array:

    if prev_x is None:
        cummulative = jnp.zeros_like(x[-1])
    else:
        cummulative = prev_x

    discounted_sum = jnp.zeros_like(x)

    i = len(x) - 1

    def _cond(carry: Tuple[int, jax.Array, jax.Array]) -> bool:
        i_, _, _ = carry
        return i_ >= 0

    def _body(carry: Tuple[int, jax.Array, jax.Array]) -> Tuple[int, jax.Array, jax.Array]:
        i_, discounted_sum_, cummulative_ = carry
        cummulative_ = x[i_] + discount * cummulative_ * (1.0 - dones[i_])
        discounted_sum_ = discounted_sum_.at[i_].set(cummulative_)
        return i_ - 1, discounted_sum_, cummulative_

    _, discounted_sum, _ = jax.lax.while_loop(
        _cond,
        _body,
        (i, discounted_sum, cummulative)
    )

    return discounted_sum


class VPPOState:
    def __init__(self, cfg):
        agent =





def train(
    state: VPPOState,
    cfg: Config,
    env: ,
    num_updates: int,
    rollout_len: int,
    key: jax.random.key,
    calculate_dcd_gradients: bool = False,
    move_mean_agent: bool = False,
    negative_measure_gradients: bool = False
):

    if calculate_dcd_gradients:
        actor = state.actors.unpack_actors()[0]
        vec_actor = VectorizedActor(actor, num_replicas=cfg.num_measures + 1, key=key)

        obs_normalizer = None
        if cfg.normalize_obs:
            obs_normalizer = actor.obs_normalizer

        returns_normalizer = None
        if cfg.normalize_returns:
            returns_normalizer = actor.returns_normalizer


    next_obs = env.reset()
    if cfg.normalize_obs:
        next_obs = state.actors.normalize_obs(next_obs)

    global_step = 0
    for u in range(1, num_updates + 1):

        rollout = make_empty_rollout(
            cfg.rollout_len,
            cfg.num_envs,
            env.obs_shape,
            env.action_shape,
            cfg.num_measures
        )

        for step in range(rollout_len):
            global_step += cfg.num_envs
            _obs = rollout.obs.at[step].set(next_obs)

            action, logprob, _ = state.actors.get_action(next_obs)
            _actions = rollout.actions.at[step].set(action)
            _logprobs = rollout.logprobs.at[step].set(logprob)

            if calculate_dcd_gradients:
                next_obs = next_obs.reshape(num_agents, cfg.num_envs // num_agents, -1)
                value = state.qd_critic.get_value(next_obs)

            elif move_mean_agent:
                value = state.mean_critic.get_value(next_obs)

            else:
                #????
                value = jnp.zeros()

            _values = rollout.values.at[step].set(value)

            next_obs, reward, dones, infos = vec_env.step(action)
            if cfg.normalize_obs:
                next_obs = state.actors.normalize_obs(next_obs)

            _truncated = rollout.truncated.at[step].set(infos["truncation"])
            _dones = rollout.dones.at[step].set(dones)

            measures = -infos["measures"] if negative_measure_gradients else infos["measures"]
            _measures = rollout.measures.at[step].set(measures)

            if move_mean_agent:
                reward_measures = jnp.stack([reward, measures], axis=1)
                reward_measures *= state._grad_coeffs
                reward = jnp.sum(reward_measures, axis=1)

            _rewards = rollout.rewards.at[step].set(reward)

            state.total_rewards += reward

            rollout = Rollout(
                obs=_obs,
                actions=_actions,
                logprobs=_logprobs,
                rewards=_rewards,
                dones=_dones,
                truncated=_truncated,
                values=_values,
                measures=_measures,
                len=rollout.len
            )

        if calculate_dcd_gradients:
            envs_pre_dim = cfg.num_envs // (cfg.num_measures + 1)
            mask = jnp.eye(cfg.num_measures + 1)

            rew_measures

            _advantages, _returns = calculate_rewards(rollout)

        else:
