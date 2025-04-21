from typing import Tuple

from gymnax.environments.environment import Environment
import jax
import jax.numpy as jnp
from flax import nnx

from viberl.algorithms.ppo._batch_update import batch_update
from viberl.algorithms._utils import normalize, policy_grad_loss, value_loss, calculate_discounted_sum
from viberl.algorithms.ppo._rollout import Rollout, make_empty_rollout
from viberl.algorithms.ppo._config import Config
from viberl.algorithms.ppo._state import PPOState


def _calculate_returns(state: PPOState, cfg: Config, rollout: Rollout) -> Tuple[jax.Array, jax.Array]:
    last_obs = rollout.obs[-1]
    next_values = state.critic(last_obs)

    if cfg.normalize_returns:
        ...

    if cfg.v_bootstrap:
        rewards = rollout.rewards + cfg.gamma * rollout.dones * rollout.truncated * rollout.values
    else:
        rewards = rollout.rewards

    _values = jnp.stack([rollout.values, next_values], axis=0)

    deltas = (rewards - _values[:-1]) + (1 - rollout.dones) * (cfg.gamma * _values[1:])
    advantages = calculate_discounted_sum(deltas, rollout.dones, cfg.gamma * cfg.gae_lambda)

    returns = advantages + _values[:-1]

    return advantages, returns

def _mb_loss(
    state: PPOState,
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

    logprob, entropy = state.actor.get_action_log_probs(b_obs, action=b_actions)

    values = state.critic(b_obs[:, mb_idxs])

    log_ratio = (logprob - b_logprobs).flatten()
    ratio = jnp.exp(log_ratio)

    old_approx_kl = jnp.mean(-log_ratio)
    approx_kl = jnp.mean(((ratio - 1.0) - log_ratio))
    clipfracs = jnp.mean((jnp.abs(ratio - 1.0) > cfg.surrogate_clip_coef).astype(jnp.float32)).item()

    mb_advantages = advantages[:, mb_idxs].flatten()
    if cfg.normalize_advantages:
        mb_advantages = normalize(mb_advantages)

    pg_loss = policy_grad_loss(mb_advantages, ratio, clip_coef=cfg.surrogate_clip_coef)
    v_loss = value_loss(
        values,
        b_values,
        returns,
        clip_coef=cfg.v_clip_coef if cfg.clip_v_loss else None
    )

    entropy_loss = entropy.mean()
    loss = pg_loss - entropy_loss * cfg.entropy_coef + v_loss * cfg.v_coef

    return loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio


@nnx.jit
def _train_step(
    state: PPOState,
    cfg: Config,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    grad_fn = nnx.value_and_grad(_mb_loss, has_aux=True)
    (loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio), grads = grad_fn(state, cfg, rollout, advantages, returns, mb_idxs)

    explained_var = 1 - jnp.var(returns - rollout.values) / jnp.var(returns)
    state.train_metrics.update( # TODO: Do we need seperate metrics?
        loss=loss,
        policy_loss=pg_loss,
        value_loss=v_loss,
        entropy=entropy_loss,
        old_approx_kl=old_approx_kl,
        approx_kl=approx_kl,
        clipfrac=clipfracs,
        explained_var=explained_var,
        ratio_min=ratio.min(),
        ratio_max=ratio.max()
    )
    # Grad clipping is part of the optimizer
    state.actor_optimizer.update(grads)
    # Grad clipping is part of the optimizer
    state.critic_optimizer.update(grads)

    return (loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio)


def train(
    state: PPOState,
    cfg: Config,
    env: Environment,
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

            action, logprob, _ = state.actor.get_action(next_obs, key=actor_key)
            _actions = rollout.actions.at[step].set(action)
            _logprobs = rollout.logprobs.at[step].set(logprob)

            value = state.critic(next_obs)
            _values = rollout.values.at[step].set(value)

            next_obs, reward, dones, infos = vec_env.step(action)
            if cfg.normalize_obs:
                next_obs = state.actor.normalize_obs(next_obs)

            _truncated = rollout.truncated.at[step].set(infos["truncation"])
            _dones = rollout.dones.at[step].set(dones)

            measures = -infos["measures"] if negative_measure_gradients else infos["measures"]
            _measures = rollout.measures.at[step].set(measures)

            reward_measures = jnp.stack([reward, measures], axis=1)
            reward_measures *= state._grad_coeffs
            reward = jnp.sum(reward_measures, axis=1)

            _rewards = rollout.rewards.at[step].set(reward)

            state.total_rewards += reward
            state.ep_len += 1

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
            train_step_fn=_train_step
        )
