from typing import Optional, Tuple, Sequence

import jax
import jax.numpy as jnp

import chex
import gymnax
from jaxlib.mlir.ir import VectorType
import numpy as np
from numpy.lib.function_base import vectorize
import optax
from flax import nnx
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

from viberl.algorithms.ppga._batch_update import batch_update
from viberl.algorithms.ppga._utils import normalize, pg_loss, v_loss, calculate_discounted_sum
from viberl.algorithms.ppga._rollout import Rollout, make_empty_rollout

from viberl.algorithms.ppga._rollout import Rollout, make_empty_rollout
from viberl.algorithms.ppga._config import Config
from viberl.algorithms.ppga._state import VPPOState


def _calculate_returns(state: VPPOState, cfg: Config, rollouts: Rollout) -> Tuple[jax.Array, jax.Array]:
    jax.lax.stop_gradient(state)
    last_obs = rollouts.obs[-1]
    next_values = state.mean_critic.get_value(last_obs)

    if cfg.normalize_returns:
        ...

    if cfg.value_bootstrap:
        rewards = rollouts.rewards + cfg.gamma * dones * rollouts.truncated * rollout.values
    else:
        rewards = rollout.rewards

    _values = jnp.stack([rollout.values, next_values], axis=0)

    deltas = (rewards - _values[:-1]) + (1 - dones) * (cfg.gamma * _values[1:])
    advantages = calculate_discounted_sum(deltas, dones, cfg.gamma * cfg.gae_lambda)

    returns = advantages + _values[:-1]

    return advantages, returns

def _mean_agent_mb_loss(
    state: VPPOState,
    cfg: Config,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    b_obs = rollout.obs[:, mb_idxs]
    b_actions = rollout.actions[:, mb_idxs]
    b_values = rollout.values[:, mb_idxs]
    b_logprobs = rollout.logprobs[:, mb_idxs]

    _, logprob, entropy = state.actors.get_action(b_obs, b_actions)

    values = state.mean_critic.get_value(b_obs[:, mb_idxs])

    log_ratio = (logprob - b_logprobs).flatten()
    ratio = jnp.exp(log_ratio)

    old_approx_kl = jnp.mean(-log_ratio)
    approx_kl = jnp.mean(((ratio - 1.0) - log_ratio))
    clipfracs = jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_coef).astype(jnp.float32)).item()

    mb_advantages = advantages[:, mb_idxs].flatten()
    if cfg.normalize_advantages:
        mb_advantages = normalize(mb_advantages)

    pg_loss = pg_loss(mb_advantages, ratio, clip_coef=cfg.clip_coef)
    v_loss = v_loss(
        values,
        b_values,
        returns,
        clip_coef=cfg.v_clip_coef if cfg.clip_v_loss else None
    )

    entropy_loss = entropy.mean()
    loss = pg_loss - entropy_loss * cfg.entropy_coef + v_loss * cfg.v_coef

    return loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio


@nnx.jit
def _mean_agent_train_step(
    state: VPPOState,
    cfg: Config,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    grad_fn = nnx.value_and_grad(_mean_agent_mb_loss, has_aux=True)
    (loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio), grads = grad_fn(state, cfg, rollout, advantages, returns, mb_idxs)
    state.metrics.update( # TODO: Do we need seperate metrics?
        loss=loss,
        pg_loss=pg_loss,
        v_loss=v_loss,
        entropy_loss=entropy_loss,
        old_approx_kl=old_approx_kl,
        approx_kl=approx_kl,
        clipfracs=clipfracs,
        ratio=ratio
    )
    # Grad clipping is part of the optimizer
    state.actor_optimizer.update(grads)
    # Grad clipping is part of the optimizer
    state.mean_critic_optimizer.update(grads)

    return (loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio)


def update_mean_agent(
    state: VPPOState,
    cfg: Config,
    env: Env,
    num_updates: int,
    rollout_len: int,
    key: jax.random.key,
    negative_measure_gradients: bool = False
):

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

            value = state.mean_critic.get_value(next_obs)
            _values = rollout.values.at[step].set(value)

            next_obs, reward, dones, infos = vec_env.step(action)
            if cfg.normalize_obs:
                next_obs = state.actors.normalize_obs(next_obs)

            _truncated = rollout.truncated.at[step].set(infos["truncation"])
            _dones = rollout.dones.at[step].set(dones)

            measures = -infos["measures"] if negative_measure_gradients else infos["measures"]
            _measures = rollout.measures.at[step].set(measures)

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
            )


        (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfrac, ratio) = batch_update(
            state,
            cfg,
            rollout,
            calculate_returns_fn=_calculate_returns,
            train_step_fn=_mean_agent_train_step
        )
