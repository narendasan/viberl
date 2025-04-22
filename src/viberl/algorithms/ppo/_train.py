from typing import Tuple

import logging
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment
import jax
import jax.numpy as jnp
from flax import nnx

from viberl.algorithms.ppo._batch_update import batch_update
from viberl.algorithms._utils import normalize, policy_grad_loss, value_loss, calculate_discounted_sum
from viberl.algorithms.ppo._rollout import Rollout, flatten_vec_rollout, make_empty_rollout
from viberl.algorithms.ppo._config import Config
from viberl.algorithms.ppo._state import PPOState
from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP

_LOGGER = logging.getLogger(__name__)

def _calculate_returns(state: PPOState, cfg: Config, rollout: Rollout) -> Tuple[jax.Array, jax.Array]:
    last_obs = rollout.obs[-1]
    next_values = state.critic(last_obs)
    collected_values = rollout.values

    if cfg.normalize_returns:
        mean, var = state.actor.returns_mean, state.actor.return_var
        next_value = (jax.lax.clamp(-5.0, next_values, 5.0) * jnp.sqrt(var)) + mean
        collected_values = (jax.lax.clamp(-5.0, collected_values, 5.0) * jnp.sqrt(var)) + mean

    if cfg.v_bootstrap:
        rewards = rollout.rewards + cfg.gamma * rollout.dones * rollout.truncated * rollout.values
    else:
        rewards = rollout.rewards

    _values = jnp.append(collected_values, jnp.expand_dims(next_values, axis=0), axis=0)

    deltas = (rewards - _values[:-1]) + (1 - rollout.dones) * (cfg.gamma * _values[1:])
    advantages = calculate_discounted_sum(deltas, rollout.dones, cfg.gamma * cfg.gae_lambda)

    returns = advantages + _values[:-1]

    return advantages, returns

def _mb_critic_loss(
    critic: CriticMLP,
    cfg: Config,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> jax.Array:
    b_obs = rollout.obs[mb_idxs, :]
    b_values = rollout.values[mb_idxs, :]
    b_returns = returns[mb_idxs, :]
    _LOGGER.debug(f"Obs minibatch: {b_obs.shape}, Value minibatch: {b_values.shape}, Returns minibatch: {b_returns.shape}")

    values = critic(b_obs)

    v_loss = value_loss(
        values,
        b_values,
        b_returns,
        clip_coef=cfg.v_clip_coef if cfg.clip_v_loss else None
    )

    return v_loss * cfg.v_coef


def _mb_actor_loss(
    actor: ActorMLP,
    cfg: Config,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:

    b_obs = rollout.obs[mb_idxs, :]
    b_actions = rollout.actions[mb_idxs, :]
    b_logprobs = rollout.logprobs[mb_idxs, :]

    _LOGGER.debug(f"Obs minibatch: {b_obs.shape}, Action minibatch: {b_actions.shape}, Logprob minibatch: {b_logprobs.shape}")

    logprob, entropy = actor.get_action_log_probs(b_obs, action=b_actions)

    log_ratio = (logprob - b_logprobs).flatten()
    ratio = jnp.exp(log_ratio)

    old_approx_kl = jnp.mean(-log_ratio)
    approx_kl = jnp.mean(((ratio - 1.0) - log_ratio))
    clipfracs = jnp.mean((jnp.abs(ratio - 1.0) > cfg.surrogate_clip_coef).astype(jnp.float32)).item()

    b_advantages = advantages[:, mb_idxs] #.flatten()
    b_returns = returns[mb_idxs, :]
    if cfg.normalize_advantages:
        mb_advantages = normalize(b_advantages)

    pg_loss = policy_grad_loss(b_advantages, ratio, clip_coef=cfg.surrogate_clip_coef)
    entropy_loss = entropy.mean()
    loss = pg_loss - entropy_loss * cfg.entropy_coef

    return loss, (pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio)



def _train_step(
    state: PPOState,
    cfg: Config,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    actor_grad_fn = nnx.value_and_grad(_mb_actor_loss, has_aux=True)
    (loss, (pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio)), actor_grads = actor_grad_fn(state.actor, cfg, rollout, advantages, returns, mb_idxs)
    explained_var = 1 - jnp.var(returns - rollout.values) / jnp.var(returns)

    critic_grad_fn = nnx.value_and_grad(_mb_critic_loss)
    v_loss, critic_grads = critic_grad_fn(state.critic, cfg, rollout, advantages, returns, mb_idxs)

    _LOGGER.info(f"Loss: {loss:.4f}, Policy Loss: {pg_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy: {entropy_loss:.4f}, Old Approx KL: {old_approx_kl:.4f}, Approx KL: {approx_kl:.4f}, Clipfrac: {clipfracs:.4f}, Explained Var: {explained_var:.4f}, Ratio Min: {ratio.min():.4f}, Ratio Max: {ratio.max():.4f}")
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
    state.actor_optimizer.update(actor_grads)
    # Grad clipping is part of the optimizer
    state.critic_optimizer.update(critic_grads)

    return loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio


def train(
    state: PPOState,
    cfg: Config,
    env_info: Tuple[Environment, EnvParams],
    key: jax.random.key,
    *,
    negative_measure_gradients: bool = False
):
    env, env_params = env_info
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    key, reset_key = jax.random.split(key, 2)
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, cfg.num_envs), env_params)

    if len(next_obs.shape) == 1:
        next_obs = jnp.expand_dims(next_obs, axis=0)

    if cfg.normalize_obs:
        next_obs = state.actor.normalize_obs(next_obs)

    num_updates = cfg.total_timesteps // (cfg.rollout_len * cfg.num_envs)
    _LOGGER.info(f"Training for {num_updates} updates")

    global_step = 0
    for u in range(1, num_updates + 1):

        rollout = make_empty_rollout(
            cfg.rollout_len,
            cfg.num_envs,
            env.observation_space(env_params).shape,
            env.action_space(env_params).shape,
        )

        _LOGGER.debug(f"Rollout: {rollout.shapes}")

        for step in range(cfg.rollout_len):
            global_step += cfg.num_envs

            _obs = rollout.obs.at[step].set(next_obs)

            key, action_key, env_step_key = jax.random.split(key, 3)

            action, logprob, _ = state.actor.get_action(next_obs, key=action_key)
            _LOGGER.debug(f"Action: {action.shape}, logprob {logprob.shape}")

            _actions = rollout.actions.at[step].set(action)
            _logprobs = rollout.logprobs.at[step].set(logprob)

            value = state.critic(next_obs)
            _LOGGER.debug(f"Value: {value.shape}")

            _values = rollout.values.at[step].set(value)

            next_obs, env_state, reward, dones, infos = vmap_step(
                jax.random.split(env_step_key, cfg.num_envs),
                env_state,
                action,
                env_params
            )

            if len(next_obs.shape) == 1:
                next_obs = jnp.expand_dims(next_obs, axis=0)

            if cfg.normalize_obs:
                next_obs = state.actor.normalize_obs(next_obs)

            _truncated = rollout.truncated.at[step].set(jnp.expand_dims(infos["truncation"], axis=-1))
            _dones = rollout.dones.at[step].set(jnp.expand_dims(dones, axis=-1))

            reward = jnp.expand_dims(reward, axis=-1)
            _LOGGER.debug(f"Reward: {reward.shape}")
            reward = jnp.sum(reward, axis=1)

            _rewards = rollout.rewards.at[step].set(jnp.expand_dims(reward, axis=-1))

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
            )

        _LOGGER.info(f"[{u}/{num_updates}] Step: {global_step} Total Reward: {state.total_rewards} Episode Len: {state.ep_len}")
        flattened_rollout = flatten_vec_rollout(rollout, env.observation_space(env_params).shape, env.action_space(env_params).shape)
        _LOGGER.debug(f"Flattened Rollout: {flattened_rollout.shapes}")
        (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfrac, ratio) = batch_update(
            state,
            cfg,
            flattened_rollout,
            calculate_returns_fn=_calculate_returns,
            train_step_fn=_train_step
        )
