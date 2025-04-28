import logging
from typing import Callable, List, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment

from viberl.algorithms._utils import (
    calculate_discounted_sum,
    normalize,
    policy_grad_loss,
    value_loss,
)
from viberl.algorithms.ppo._config import Config, _TrainingSettingConfigSubset
from viberl.algorithms.ppo._eval import eval
from viberl.algorithms.ppo._rollout import (
    Rollout,
    flatten_vec_rollout,
    make_empty_rollout,
)
from viberl.algorithms.ppo._state import PPOState
from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP

_LOGGER = logging.getLogger(__name__)

train_fn = Callable[[PPOState, Config, Rollout, jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array, jax.Array,  jax.Array, jax.Array, jax.Array, jax.Array]]

returns_fn = Callable[[PPOState, Config, Rollout], Tuple[jax.Array, jax.Array]]

#@nnx.jit
def _calculate_returns(state: PPOState, cfg: _TrainingSettingConfigSubset, rollout: Rollout) -> Tuple[jax.Array, jax.Array]:
    last_obs = rollout.obs[-1]
    next_values = state.critic(last_obs)
    collected_values = rollout.values

    next_values = nnx.cond(
        cfg.normalize_returns,
        lambda: (jax.lax.clamp(-5.0, next_values, 5.0) * jnp.sqrt(state.actor.returns_var)) +  state.actor.returns_mean,
        lambda: next_values
    )

    collected_values = nnx.cond(
        cfg.normalize_returns,
        lambda: (jax.lax.clamp(-5.0, collected_values, 5.0) * jnp.sqrt(state.actor.returns_var)) +  state.actor.returns_mean,
        lambda: collected_values
    )

    rewards = nnx.cond(
        cfg.v_bootstrap,
        lambda: rollout.rewards + cfg.gamma * rollout.dones * rollout.truncated * next_values,
        lambda: rollout.rewards
    )

    _values = jnp.append(collected_values, jnp.expand_dims(next_values, axis=0), axis=0)

    deltas = (rewards - _values[:-1]) + (1 - rollout.dones) * (cfg.gamma * _values[1:])
    advantages = calculate_discounted_sum(deltas, rollout.dones, cfg.gamma * cfg.gae_lambda)

    returns = advantages + _values[:-1]

    return advantages, returns

@nnx.jit
def _mb_critic_loss(
    critic: CriticMLP,
    cfg: _TrainingSettingConfigSubset,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> jax.Array:
    b_obs = rollout.obs[mb_idxs, :]
    b_values = rollout.values[mb_idxs, :]
    b_returns = returns[mb_idxs, :]
    #_LOGGER.debug(f"Obs minibatch: {b_obs.shape}, Value minibatch: {b_values.shape}, Returns minibatch: {b_returns.shape}")

    values = critic(b_obs)

    v_loss = nnx.cond(
        cfg.clip_v_loss,
        lambda: value_loss(values, b_values, b_returns, clip_coef=cfg.v_clip_coef),
        lambda: value_loss(values, b_values, b_returns, clip_coef=None)
    )

    return v_loss * cfg.v_coef


@nnx.jit
def _mb_actor_loss(
    actor: ActorMLP,
    cfg: _TrainingSettingConfigSubset,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:

    b_obs = rollout.obs[mb_idxs, :]
    b_actions = rollout.actions[mb_idxs, :]
    b_logprobs = rollout.logprobs[mb_idxs, :]

    #_LOGGER.debug(f"Obs minibatch: {b_obs.shape}, Action minibatch: {b_actions.shape}, Logprob minibatch: {b_logprobs.shape}")

    logprob, entropy = actor.get_action_log_probs(b_obs, action=b_actions)

    log_ratio = (logprob - b_logprobs).flatten()
    ratio = jnp.exp(log_ratio)

    old_approx_kl = jnp.mean(-log_ratio)
    approx_kl = jnp.mean(((ratio - 1.0) - log_ratio))
    clipfracs = jnp.mean((jnp.abs(ratio - 1.0) > cfg.surrogate_clip_coef).astype(jnp.float32)) #.item()

    b_advantages = advantages[:, mb_idxs] #.flatten()
    b_returns = returns[mb_idxs, :]
    mb_advantages = nnx.cond(cfg.normalize_advantages, lambda: normalize(b_advantages), lambda: b_advantages)

    pg_loss = policy_grad_loss(b_advantages, ratio, clip_coef=cfg.surrogate_clip_coef)
    entropy_loss = entropy.mean()
    loss = pg_loss - entropy_loss * cfg.entropy_coef

    return loss, (pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio)


@nnx.jit
def _train_step(
    state: PPOState,
    cfg: _TrainingSettingConfigSubset,
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
    ratio_min = ratio.min()
    ratio_max = ratio.max()

    # jax.experimental.io_callback(
    #     lambda loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var, ratio_min, ratio_max: _LOGGER.debug(f"Loss: {loss}, Policy Loss: {pg_loss}, Value Loss: {v_loss}, Entropy: {entropy_loss}, Old Approx KL: {old_approx_kl}, Approx KL: {approx_kl}, Clipfrac: {clipfracs}, Explained Var: {explained_var}, Ratio Min: {ratio_min}, Ratio Max: {ratio_max}"),
    #     None,
    #     loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var, ratio_min, ratio_max
    # )

    state.train_metrics.update( # TODO: Do we need seperate metrics?
        loss=loss,
        policy_loss=pg_loss,
        value_loss=v_loss,
        entropy=entropy_loss,
        old_approx_kl=old_approx_kl,
        approx_kl=approx_kl,
        clipfrac=clipfracs,
        explained_var=explained_var,
        ratio_min=ratio_min,
        ratio_max=ratio_max
    )
    # Grad clipping is part of the optimizer
    state.actor_optimizer.update(actor_grads)
    # Grad clipping is part of the optimizer
    state.critic_optimizer.update(critic_grads)

    return loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio

def batch_update(
    state: PPOState,
    cfg: _TrainingSettingConfigSubset,
    rollout: Rollout,
    *,
    calculate_returns_fn: returns_fn,
    train_step_fn: train_fn) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, List[jax.Array], jax.Array]:

    advantages, returns = calculate_returns_fn(state, cfg, rollout)

    if cfg.normalize_returns:
        returns = state.actor.normalize_returns(returns)

    batch_size = rollout.obs.shape[0]
    minibatch_size = batch_size // cfg.num_minibatches
    #_LOGGER.debug(f"Batch size: {batch_size}, minibatch size: {minibatch_size}")

    batch_idxs = jnp.arange(batch_size)
    clipfracs: List[jax.Array] = []

    pg_loss = v_loss = entropy_loss = ratio = old_approx_kl = approx_kl = jnp.empty(batch_size)

    # TODO: JAXIFY
    for epoch in range(cfg.num_update_epochs):
        for mb_start in range(0, batch_size, cfg.num_minibatches):
            mb_end = mb_start + minibatch_size

            mb_idxs = batch_idxs[mb_start : mb_end]
            (loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, _clipfracs, ratio) = train_step_fn(state, cfg, rollout, advantages, returns, mb_idxs)
            clipfracs.append(_clipfracs)

        if cfg.target_kl:
            if approx_kl < cfg.target_kl:
                _LOGGER.info(f"Achieved target KL divergance, stopping early at epoch {epoch}")


    ratio_min = ratio.min()
    ratio_max = ratio.max()

    # jax.experimental.io_callback(
    #     lambda pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio_min, ratio_max: _LOGGER.debug(f"Policy Loss: {pg_loss}, Value Loss: {v_loss}, Entropy: {entropy_loss}, Old Approx KL: {old_approx_kl}, Approx KL: {approx_kl}, Clipfrac: {clipfracs}, Ratio Min: {ratio_min}, Ratio Max: {ratio_max}"),
    #     None,
    #     pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio_min, ratio_max
    # )

    return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, ratio

def train(
    state: PPOState,
    cfg: Config,
    env_info: Tuple[Environment, EnvParams],
    key: jax.random.key,
    *,
    eval_callback: Optional[Callable[[PPOState, Config, jax.Array], None]] = None,
    negative_measure_gradients: bool = False
):
    training_cfg = cfg.training_config_subset()

    total_rewards = jnp.zeros((cfg.num_envs,))
    ep_len = jnp.zeros((cfg.num_envs,))

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
    #_LOGGER.info(f"Training for {num_updates} updates")

    global_step = 0
    for u in range(1, num_updates + 1):

        rollout = make_empty_rollout(
            cfg.rollout_len,
            cfg.num_envs,
            env.observation_space(env_params).shape,
            env.action_space(env_params).shape,
        )

        #_LOGGER.debug(f"Rollout: {rollout.shapes}")

        def _rollout_step(
            step: int,
            carry: Tuple[Rollout, jax.Array, jax.Array, int, jax.Array, jax.Array, jax.Array]
        ) -> Tuple[Rollout, jax.Array, jax.Array, int, jax.Array, jax.Array, jax.Array]:
            (rollout, next_obs, env_state, global_step, total_rewards, ep_len, key) = carry
            global_step += cfg.num_envs

            _obs = rollout.obs.at[step].set(next_obs)

            key, action_key, env_step_key = jax.random.split(key, 3)

            action, logprob, _ = state.actor.get_action(next_obs, key=action_key)
            #_LOGGER.debug(f"Action: {action.shape}, logprob {logprob.shape}")

            _actions = rollout.actions.at[step].set(action)
            _logprobs = rollout.logprobs.at[step].set(logprob)

            value = state.critic(next_obs)
            #_LOGGER.debug(f"Value: {value.shape}")

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
            #_LOGGER.debug(f"Reward: {reward.shape}")
            reward = jnp.sum(reward, axis=1)

            _rewards = rollout.rewards.at[step].set(jnp.expand_dims(reward, axis=-1))

            total_rewards += reward
            ep_len += 1

            rollout = Rollout(
                obs=_obs,
                actions=_actions,
                logprobs=_logprobs,
                rewards=_rewards,
                dones=_dones,
                truncated=_truncated,
                values=_values,
            )

            return rollout, next_obs, env_state, global_step, total_rewards, ep_len, key

        (rollout, next_obs, env_state, global_step, total_rewards, ep_len, key) = jax.lax.fori_loop(
            0,
            cfg.rollout_len,
            _rollout_step,
            (rollout, next_obs, env_state, global_step, total_rewards, ep_len, key)
        )

        jax.experimental.io_callback(
            lambda u, num_updates, global_step, total_rewards, ep_len: _LOGGER.info(f"[{u}/{num_updates}] Step: {global_step} Total Reward: {total_rewards} Episode Len: {ep_len}"),
            None,
            u, num_updates, global_step, total_rewards, ep_len
        )
        flattened_rollout = flatten_vec_rollout(rollout, env.observation_space(env_params).shape, env.action_space(env_params).shape)
        #_LOGGER.debug(f"Flattened Rollout: {flattened_rollout.shapes}")
        (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfrac, ratio) = batch_update(
            state,
            training_cfg,
            flattened_rollout,
            calculate_returns_fn=_calculate_returns,
            train_step_fn=_train_step
        )

        if eval_callback:
            if u % cfg.eval_frequency == 0:
                eval(state, cfg, env, key, collect_values=False, eval_callback=eval_callback)
