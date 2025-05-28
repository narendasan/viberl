import logging
from typing import Callable, Tuple, Optional, Dict, Any

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment

from viberl.algorithms._utils import (
    normalize,
    policy_grad_loss,
    value_loss,
)
from viberl.algorithms.ppga._config import Config, _TrainingConfig
from viberl.algorithms.ppga._eval import eval
from viberl.algorithms.ppga._rollout import (
    Rollout,
    flatten_vec_rollout,
    make_empty_rollout,
)
from viberl.algorithms.ppga._state import State, VecActorQDCritic, TrainState, make_train_state
from viberl.algorithms.ppga._batch_update import batch_update, train_fn, returns_fn
from viberl.models._actor import VectorizedActor
from viberl.models._critic import QDCritic
from viberl.utils.types import EvalCallback, PolicyEvalResult
from viberl.utils._eval_callbacks import _default_eval_callback


_LOGGER = logging.getLogger(__name__)


def _calculate_gae(state: State, cfg: _TrainingConfig, rollout: Rollout) -> Tuple[jax.Array, jax.Array]:
    last_obs = rollout.obs[-1, :]
    last_value = state.actor_critic.critic(last_obs)
    last_done = rollout.dones[-1, :]
    #last_value = jnp.where(rollout.dones[-1, :], 0, last_value)

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
    critic: QDCritic,
    cfg: _TrainingConfig,
    rollout: Rollout,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> jax.Array:
    b_obs = rollout.obs[mb_idxs, :]
    b_values = rollout.values[mb_idxs, :]
    b_returns = returns[mb_idxs, :]
    #_LOGGER.debug(f"Obs minibatch: {b_obs.shape}, Value minibatch: {b_values.shape}, Returns minibatch: {b_returns.shape}")

    values = critic.get_all_values(b_obs)

    v_loss = nnx.cond(
        cfg.clip_v_loss,
        lambda: value_loss(values, b_values, b_returns, clip_coef=cfg.v_clip_coef, clip=True),
        lambda: value_loss(values, b_values, b_returns, clip=False)
    )

    return v_loss * cfg.v_coef


@nnx.jit
def _mb_actor_loss(
    actor: VectorizedActor,
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

    logprob, entropy = actor.get_action_log_probs(b_obs, actions=b_actions)

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
    actor_critic: VecActorQDCritic,
    cfg: _TrainingConfig,
    rollout: Rollout,
    advantages: jax.Array,
    returns: jax.Array,
    mb_idxs: jax.Array
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    pg_loss, entropy_loss, approx_kl, clipfracs, ratio = _mb_actor_loss(actor_critic.actor, cfg, rollout, advantages, mb_idxs)
    value_loss = _mb_critic_loss(actor_critic.critic, cfg, rollout, returns, mb_idxs)
    loss = pg_loss + entropy_loss * cfg.entropy_coef + value_loss * cfg.v_coef
    return loss, (pg_loss, value_loss, entropy_loss, approx_kl, clipfracs, ratio)


@nnx.scan(in_axes=(nnx.Carry, None, None, None, None, 0), out_axes=(nnx.Carry, 0, 0, 0, 0, 0, 0, 0))
def _train_epoch(
    state: TrainState,
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
        ratio_max=ratio_max
    )
    # Grad clipping is part of the optimizer
    state.optimizer.update(grads)

    return state, loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio

def calculate_dcd_gradients(
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

    vec_actor = VectorizedActor(state.actor, cfg.num_measures + 1, key=key)
    actor_critic = VecActorQDCritic(vec_actor, state.qd_critic)
    train_state = make_train_state(actor_critic, cfg)

    if len(next_obs.shape) == 1:
        next_obs = jnp.expand_dims(next_obs, axis=0)

    if cfg.normalize_obs:
        next_obs = train_state.model.actor.normalize_obs(next_obs)

    if eval_callback is None:
        eval_callback = _default_eval_callback

    num_train_per_eval = cfg.eval_frequency // (cfg.rollout_len * cfg.num_envs)
    assert num_train_per_eval > 0, "Eval Frequency must be >= Rollout Len * Num Envs"
    num_updates = cfg.total_timesteps //  cfg.rollout_len * cfg.num_envs * num_train_per_eval
    _LOGGER.info(f"Training for {num_updates} updates")

    train_state.global_step = 0
    def _train_eval(
        e: int,
        train_eval_carry: Tuple[TrainState, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> Tuple[TrainState, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]:
        train_state, env_state, next_obs, total_rewards, ep_len, key = train_eval_carry
        def _train(
            t: int,
            train_carry: Tuple[TrainState, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]
        ) -> Tuple[TrainState, Dict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array]:
            train_state, env_state, next_obs, total_rewards, ep_len, key = train_carry
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
                    rollout_carry: Tuple[TrainState, Rollout, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
                ) -> Tuple[TrainState, Rollout, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                    (train_state, rollout, next_obs, env_state, total_rewards, ep_len, key) = rollout_carry
                    train_state.global_step += cfg.num_envs

                    _obs = rollout.obs.at[step].set(next_obs)

                    key, action_key, env_step_key = jax.random.split(key, 3)

                    action, logprob, _ = train_state.model.actor.get_action(next_obs, key=action_key)
                    #_LOGGER.debug(f"Action: {action.shape}, logprob {logprob.shape}")
                    # if discrete:
                        # action =
                    # else

                    _actions = rollout.actions.at[step].set(action)
                    _logprobs = rollout.logprobs.at[step].set(logprob)

                    #_LOGGER.debug(f"Value: {value.shape}")

                    value = train_state.model.critic(next_obs)
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
                        # Need this indirection so that state can be mutated in ActorMLP
                        next_obs = train_state.model.actor.normalize_obs(next_obs)

                    _truncated = rollout.truncated.at[step].set(jnp.expand_dims(infos["truncation"], axis=-1))
                    _dones = rollout.dones.at[step].set(jnp.expand_dims(dones, axis=-1))

                    if cfg.normalize_rewards:
                        reward = train_state.model.actor.normalize_rewards(reward)

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

                    return train_state, rollout, next_obs, env_state, total_rewards, ep_len, key

                (train_state, rollout, next_obs, env_state, total_rewards, ep_len, key) = nnx.fori_loop(
                    0,
                    cfg.rollout_len,
                    _rollout_step,
                    (train_state, rollout, next_obs, env_state, total_rewards, ep_len, key)
                )

            train_state.rollout_metrics.update(train_reward=total_rewards)

            flattened_rollout = flatten_vec_rollout(rollout, env.observation_space(env_params).shape, env.action_space(env_params).shape)
            #_LOGGER.debug(f"Flattened Rollout: {flattened_rollout.shapes}")

            with jax.profiler.TraceAnnotation("training_loop"):
                (state, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, ratio) = batch_update(
                    train_state,
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

        eval_result, eval_rollout = eval(train_state, eval_cfg, env_info[1], vmap_reset, vmap_step, key, collect_values=False)
        train_state.eval_metrics.update(
            reward=eval_result.returns,
            ep_len=eval_result.lengths,
        )
        eval_callback(train_state, cfg, eval_result, eval_rollout)
        train_state.train_metrics.reset()
        train_state.eval_metrics.reset()

        (state, env_state, next_obs, total_rewards, ep_len, key) = nnx.fori_loop(
            0,
            num_train_per_eval,
            _train,
            (train_state, env_state, next_obs, jnp.zeros_like(total_rewards), ep_len, key)
        )

        return train_state, env_state, next_obs, total_rewards, ep_len, key

    train_state, env_state, next_obs, total_rewards, ep_len, key = nnx.fori_loop(
        0,
        num_updates,
        _train_eval,
        (train_state, env_state, next_obs, total_rewards, ep_len, key)
    )

    eval_result, eval_rollout = eval(train_state, eval_cfg, env_info[1], vmap_reset, vmap_step, key, collect_values=False)
    train_state.eval_metrics.update(
        train_reward=total_rewards,
        reward=eval_result.returns,
        ep_len=eval_result.lengths,
    )
    eval_callback(train_state, cfg, eval_result, eval_rollout)
    train_state.train_metrics.reset()
    train_state.eval_metrics.reset()

    state.actor = train_state.model.actor.unpack_actors()[0]
    return state, eval_result
