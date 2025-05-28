import logging
from typing import Callable, Tuple, Optional, Dict, Any

import jax
import jax.numpy as jnp
from flax import nnx
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment

from viberl.algorithms._utils import (
    normalize,
    policy_grad_loss,
    value_loss,
)
from viberl.algorithms.ppo._config import Config, _TrainingConfig
from viberl.algorithms.ppo._eval import eval
from viberl.algorithms.ppo._rollout import (
    Rollout,
    flatten_vec_rollout,
    make_empty_rollout,
)
from viberl.algorithms.ppo._state import State, ActorCritic
from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP
from viberl.utils.types import EvalCallback, PolicyEvalResult
from viberl.utils._eval_callbacks import _default_eval_callback

_LOGGER = logging.getLogger(__name__)

train_fn = Callable[[State, _TrainingConfig, Rollout, jax.Array, jax.Array, jax.Array], Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array,  jax.Array, jax.Array, jax.Array]]
returns_fn = Callable[[State, _TrainingConfig, Rollout], Tuple[jax.Array, jax.Array]]

def _calculate_gae(state: State, cfg: _TrainingConfig, rollout: Rollout) -> Tuple[jax.Array, jax.Array]:
    last_obs = rollout.obs[-1, :]
    last_value = state.actor_critic.critic(last_obs)
    last_done = rollout.dones[-1, :]
    #last_value = jnp.where(rollout.dones[-1, :], 0, last_value)
    if cfg.v_bootstrap:
        rollout.rewards = rollout.rewards + cfg.gamma * rollout.dones * rollout.truncated * rollout.values

    def _calculate_advantages(carry: Tuple[jax.Array, jax.Array, jax.Array], r: Rollout) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        next_advantage, next_value, next_done = carry
        delta = r.rewards.squeeze() + cfg.gamma * next_value * (1 - next_done) - r.values
        advantage = delta + cfg.gamma * cfg.gae_lambda * (1 - next_done) * next_advantage
        jax.experimental.io_callback(
            lambda r: _LOGGER.debug(f"reward: {r.rewards}, value: {r.values}, r: {r}"),
            None,
            r
        )
        return (advantage, r.values, r.dones), advantage

    _, advantages = jax.lax.scan(
        _calculate_advantages,
        (jnp.zeros_like(last_value), last_value, last_done),
        rollout,
        reverse=True
    )

    return advantages, advantages + rollout.values

@nnx.jit
def _mb_critic_loss(
    critic: CriticMLP,
    cfg: _TrainingConfig,
    rollout: Rollout,
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
        lambda: value_loss(values, b_values, b_returns, clip_coef=cfg.v_clip_coef, clip=True),
        lambda: value_loss(values, b_values, b_returns, clip=False)
    )

    return v_loss * cfg.v_coef


@nnx.jit
def _mb_actor_loss(
    actor: ActorMLP,
    cfg: _TrainingConfig,
    rollout: Rollout,
    advantages: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    # jax.experimental.io_callback(
    #     lambda idxs: _LOGGER.debug(f"idxs: {idxs}"),
    #     None,
    #     mb_idxs
    # )

    b_obs = rollout.obs[mb_idxs, :]
    b_actions = rollout.actions[mb_idxs, :]
    b_logprobs = rollout.logprobs[mb_idxs, :]
    b_advantages = advantages[mb_idxs, :] #.flatten()
    jax.experimental.io_callback(
        lambda r: _LOGGER.debug(f"advantages: {r},"),
        None,
        b_advantages
    )

    #_LOGGER.debug(f"Obs minibatch: {b_obs.shape}, Action minibatch: {b_actions.shape}, Logprob minibatch: {b_logprobs.shape}")

    logprob, entropy = actor.get_action_log_probs(b_obs, action=b_actions)

    log_ratio = (logprob - b_logprobs).flatten()
    ratio = jnp.exp(log_ratio)

    approx_kl = jnp.mean(((ratio - 1.0) - log_ratio))
    clipfracs = jnp.mean((jnp.abs(ratio - 1.0) > cfg.surrogate_clip_coef).astype(jnp.float32)) #.item()

    mb_advantages = nnx.cond(cfg.normalize_advantages, lambda: normalize(b_advantages), lambda: b_advantages)

    pg_loss = policy_grad_loss(mb_advantages, ratio, clip_coef=cfg.surrogate_clip_coef)
    entropy_loss = entropy.mean()

    return pg_loss, entropy_loss, approx_kl, clipfracs, ratio

@nnx.jit
def mb_loss(
    actor_critic: ActorCritic,
    cfg: _TrainingConfig,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    pg_loss, entropy_loss, approx_kl, clipfracs, ratio = _mb_actor_loss(actor_critic.actor, cfg, rollout, advantages, mb_idxs)
    value_loss = _mb_critic_loss(actor_critic.critic, cfg, rollout, returns, mb_idxs)
    loss = pg_loss + entropy_loss * cfg.entropy_coef + value_loss * cfg.v_coef
    return loss, (pg_loss, value_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), jax.lax.stop_gradient(clipfracs), jax.lax.stop_gradient(ratio))


@nnx.scan(in_axes=(nnx.Carry, None, None, None, None, 0), out_axes=(nnx.Carry, 0, 0, 0, 0, 0, 0, 0))
def _train_epoch(
    state: State,
    cfg: _TrainingConfig,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Scans over mb_idxs to train a full epoch per call
    """

    grad_fn = nnx.value_and_grad(mb_loss, has_aux=True)
    (loss, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)), grads = grad_fn(state.actor_critic, cfg, rollout, advantages, returns, mb_idxs)
    explained_var = 1 - jnp.var(returns - rollout.values) / jnp.var(returns)

    ratio_min = ratio.min()
    ratio_max = ratio.max()

    jax.experimental.io_callback(
        lambda loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, explained_var, ratio_min, ratio_max: _LOGGER.debug(f"Loss: {loss}, Policy Loss: {pg_loss}, Value Loss: {v_loss}, Entropy: {entropy_loss}, Approx KL: {approx_kl}, Clipfrac: {clipfracs}, Explained Var: {explained_var}, Ratio Min: {ratio_min}, Ratio Max: {ratio_max}"),
        None,
        loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, explained_var, ratio_min, ratio_max
    )

    state.train_metrics.update( # TODO: Do we need seperate metrics?
        loss=loss,
        policy_loss=pg_loss,
        value_loss=v_loss,
        entropy=entropy_loss,
        approx_kl=approx_kl,
        clipfrac=clipfracs,
        explained_var=explained_var,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        lr=state.optimizer.opt_state.hyperparams['lr_schedule']
    )
    # Grad clipping is part of the optimizer
    state.optimizer.update(grads)

    return state, loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio

def batch_update(
    state: State,
    cfg: _TrainingConfig,
    rollout: Rollout,
    *,
    calculate_returns_fn: returns_fn,
    train_step_fn: train_fn
) -> Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    @nnx.jit(static_argnums=(1,))
    def _batch_update(
        state: State,
        cfg: _TrainingConfig,
        rollout: Rollout
    ) -> Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        advantages, returns = jax.lax.stop_gradient(calculate_returns_fn(state, cfg, rollout))

        batch_size = rollout.obs.shape[0]
        minibatch_size = batch_size // cfg.num_minibatches

        batch_idxs = jnp.arange(batch_size)
        mb_idxs = batch_idxs.reshape(-1, minibatch_size)

        pg_loss = v_loss = entropy_loss = ratio = approx_kl = clipfracs = jnp.empty((cfg.num_update_epochs, cfg.num_minibatches))
        ratio = jnp.empty((cfg.num_update_epochs, cfg.num_minibatches, minibatch_size))

        # TODO: JAXIFY
        def _update_epoch(
            e: int,
            carry: Tuple[State, _TrainingConfig, Rollout, jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]
        ) -> Tuple[State, _TrainingConfig, Rollout, jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array,jax.Array, jax.Array]]:
            state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio) = carry
            state, _, _pg_loss, _v_loss, _entropy_loss, _approx_kl, _clipfracs, _ratio = train_step_fn(state, cfg, rollout, advantages, returns, mb_idxs)
            pg_loss = pg_loss.at[e].set(_pg_loss)
            v_loss = v_loss.at[e].set(_v_loss)
            entropy_loss = entropy_loss.at[e].set(_entropy_loss)
            ratio = ratio.at[e].set(_ratio)
            approx_kl = approx_kl.at[e].set(_approx_kl)
            clipfracs = clipfracs.at[e].set(_clipfracs)
            return state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)

        (state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)) = nnx.fori_loop(
            0,
            cfg.num_update_epochs,
            _update_epoch,
            (state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)),
        )


        ratio_min = ratio[-1][-1].min()
        ratio_max = ratio[-1][-1].max()

        return state, pg_loss[-1][-1], v_loss[-1][-1], entropy_loss[-1][-1], approx_kl[-1][-1], clipfracs[-1], ratio[-1][-1]
    return _batch_update(state, cfg, rollout)


def train(
    state: State,
    cfg: Config,
    env_info: Tuple[Environment, EnvParams],
    key: jax.random.key,
    *,
    eval_callback: Optional[EvalCallback] = None,
) -> Tuple[State, PolicyEvalResult]:
    training_cfg = cfg.training_config_subset()
    eval_cfg = cfg.eval_config_subset()

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
        next_obs = state.actor_critic.actor.normalize_obs(next_obs)

    if eval_callback is None:
        eval_callback = _default_eval_callback

    num_train_per_eval = cfg.eval_frequency // (cfg.rollout_len * cfg.num_envs)
    assert num_train_per_eval > 0, "Eval Frequency must be >= Rollout Len * Num Envs"
    num_updates = (cfg.total_timesteps //  (cfg.rollout_len * cfg.num_envs)) // num_train_per_eval

    _LOGGER.info(f"Training for {num_updates} updates")

    state.global_step = 0
    def _train_eval(
        e: int,
        train_eval_carry: Tuple[State, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> Tuple[State, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]:
        state, env_state, next_obs, total_rewards, ep_len, key = train_eval_carry
        def _train(
            t: int,
            train_carry: Tuple[State, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]
        ) -> Tuple[State, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]:
            state, env_state, next_obs, total_rewards, ep_len, key = train_carry
            with jax.profiler.TraceAnnotation("collect_rollout"):
                rollout = make_empty_rollout(
                    cfg.rollout_len,
                    cfg.num_envs,
                    env.observation_space(env_params).shape,
                    env.action_space(env_params).shape,
                )

                #_LOGGER.debug(f"Rollout: {rollout.shapes}")

                @nnx.jit
                def _rollout_step(
                    step: int,
                    rollout_carry: Tuple[State, Rollout, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
                ) -> Tuple[State, Rollout, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                    (state, rollout, next_obs, env_state, total_rewards, ep_len, key) = rollout_carry
                    state.global_step += cfg.num_envs

                    _obs = rollout.obs.at[step].set(next_obs)

                    key, action_key, env_step_key = jax.random.split(key, 3)

                    action, logprob, _ = state.actor_critic.actor.get_action(next_obs, key=action_key)
                    #_LOGGER.debug(f"Action: {action.shape}, logprob {logprob.shape}")
                    # if discrete:
                        # action =
                    # else

                    _actions = rollout.actions.at[step].set(action)
                    _logprobs = rollout.logprobs.at[step].set(logprob)

                    #_LOGGER.debug(f"Value: {value.shape}")

                    value = state.actor_critic.critic(next_obs)
                    _values = rollout.values.at[step].set(value)

                    next_obs, env_state, reward, dones, infos = vmap_step(
                        jax.random.split(env_step_key, cfg.num_envs),
                        env_state,
                        action,
                        env_params
                    )

                    dones = jnp.logical_or(dones, infos["truncation"])

                    if len(next_obs.shape) == 1:
                        next_obs = jnp.expand_dims(next_obs, axis=0)

                    if cfg.normalize_obs:
                        # Need this indirection so that state can be mutated in ActorMLP
                        next_obs = state.actor_critic.actor.normalize_obs(next_obs)

                    _truncated = rollout.truncated.at[step].set(jnp.expand_dims(infos["truncation"], axis=-1))
                    _dones = rollout.dones.at[step].set(jnp.expand_dims(dones, axis=-1))

                    if cfg.normalize_rewards:
                        reward = state.actor_critic.actor.normalize_rewards(reward)

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

                    total_rewards = total_rewards.at[jnp.where(dones == 1, 1, dones)].set(0)
                    ep_len = ep_len.at[jnp.where(dones == 1, 1, dones)].set(0)

                    return state, rollout, next_obs, env_state, total_rewards, ep_len, key

                (state, rollout, next_obs, env_state, total_rewards, ep_len, key) = nnx.fori_loop(
                    0,
                    cfg.rollout_len,
                    _rollout_step,
                    (state, rollout, next_obs, env_state, total_rewards, ep_len, key)
                )

            state.rollout_metrics.update(train_reward=total_rewards)

            flattened_rollout = flatten_vec_rollout(rollout, env.observation_space(env_params).shape, env.action_space(env_params).shape)
            #_LOGGER.debug(f"Flattened Rollout: {flattened_rollout.shapes}")

            with jax.profiler.TraceAnnotation("training_loop"):
                (state, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, ratio) = batch_update(
                    state,
                    training_cfg,
                    flattened_rollout,
                    calculate_returns_fn=_calculate_gae,
                    train_step_fn=_train_epoch
                )

            # if u % cfg.eval_frequency == 0:
            #     eval_result, eval_rollout = eval(state, cfg, env_info, key, collect_values=False)
            #     if eval_callback:
            #         eval_callback(state, cfg, eval_result, eval_rollout)
            #     state.train_metrics.reset()
            #     state.eval_metrics.reset()

            return state, env_state, next_obs, total_rewards, ep_len, key

        eval_result, eval_rollout = eval(state, eval_cfg, env_info[1], vmap_reset, vmap_step, key, collect_values=False)
        state.eval_metrics.update(
            reward=eval_result.returns,
            ep_len=eval_result.lengths,
        )
        eval_callback(state, cfg, eval_result, eval_rollout)
        state.train_metrics.reset()
        state.eval_metrics.reset()

        (state, env_state, next_obs, total_rewards, ep_len, key) = nnx.fori_loop(
            0,
            num_train_per_eval,
            _train,
            (state, env_state, next_obs, jnp.zeros_like(total_rewards), ep_len, key)
        )

        return state, env_state, next_obs, total_rewards, ep_len, key

    state, env_state, next_obs, total_rewards, ep_len, key = nnx.fori_loop(
        0,
        num_updates,
        _train_eval,
        (state, env_state, next_obs, total_rewards, ep_len, key)
    )

    eval_result, eval_rollout = eval(state, eval_cfg, env_info[1], vmap_reset, vmap_step, key, collect_values=False)
    state.eval_metrics.update(
        train_reward=total_rewards,
        reward=eval_result.returns,
        ep_len=eval_result.lengths,
    )
    eval_callback(state, cfg, eval_result, eval_rollout)
    state.train_metrics.reset()
    state.eval_metrics.reset()
    return state, eval_result
