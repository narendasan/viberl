import logging
from typing import Callable, Optional, Tuple, Dict, Any

from flax import nnx
import jax
import jax.numpy as jnp
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment


from viberl.utils.types import PolicyEvalResult
from viberl.algorithms.ppo._config import _EvalConfig
from viberl.algorithms.ppo._rollout import Rollout, make_empty_rollout
from viberl.algorithms.ppo._state import State

_LOGGER = logging.getLogger(__name__)

#@nnx.jit(static_argnums=(1,2,3,4,6))
def eval(
    state: State,
    cfg: _EvalConfig,
    env_params: EnvParams,
    vmap_reset: Callable,
    vmap_step: Callable,
    key: jax.Array,
    collect_values: bool,
) -> Tuple[PolicyEvalResult, Rollout]:

    total_rewards = jnp.zeros((cfg.num_envs,))
    ep_len = jnp.zeros((cfg.num_envs,))
    dones = jnp.zeros((cfg.num_envs,), dtype=bool)

    key, reset_key = jax.random.split(key, 2)
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, cfg.num_envs), env_params)

    obs_mean, obs_var = state.actor.obs_mean.value, state.actor.obs_var.value

    if len(next_obs.shape) == 1:
        next_obs = jnp.expand_dims(next_obs, axis=0)

    if cfg.normalize_obs:
        next_obs = (next_obs - obs_mean) / (jnp.sqrt(obs_var) + 1e-8)

    rollout = make_empty_rollout(
        cfg.rollout_len,
        cfg.num_envs,
        state.actor.obs_shape,
        state.actor.action_shape,
    )

    _LOGGER.debug(f"Rollout: {rollout.shapes}")

    step = 0

    def _cond_fn(carry: Tuple[State, Rollout, Dict[str, Any], jax.Array, jax.Array, int, jax.Array, jax.Array, jax.Array]) -> bool:
        state, rollout, env_state, next_obs, dones, step, total_rewards, ep_len, key = carry
        return jnp.logical_and(
            jnp.all(step < cfg.rollout_len), jnp.logical_not(jnp.all(dones))
        )

    def _body_fn(
        carry: Tuple[State, Rollout, Dict[str, Any], jax.Array, jax.Array, int, jax.Array, jax.Array, jax.Array]
    ) -> Tuple[State, Rollout, Dict[str, Any], jax.Array, jax.Array, int, jax.Array, jax.Array, jax.Array]:
        state, rollout, env_state, next_obs, dones, step, total_rewards, ep_len, key = carry
        key, action_key, env_step_key = jax.random.split(key, 3)

        _obs = rollout.obs.at[step].set(next_obs)
        action, logprob, _ = state.actor.get_action(next_obs, key=action_key)
        _LOGGER.debug(f"Action: {action.shape}, logprob {logprob.shape}")

        _actions = rollout.actions.at[step].set(action)
        _logprobs = rollout.logprobs.at[step].set(logprob)

        def _compute_values(next_obs: jax.Array) -> jax.Array:
            value = state.critic(next_obs)
            _LOGGER.debug(f"Value: {value.shape}")
            return rollout.values.at[step].set(value)

        _values = nnx.cond(
            collect_values,
            _compute_values,
            lambda next_obs: rollout.values,
            next_obs
        )

        next_obs, env_state, reward, next_dones, infos = vmap_step(
            jax.random.split(env_step_key, cfg.num_envs),
            env_state,
            action,
            env_params
        )

        if len(next_obs.shape) == 1:
            next_obs = jnp.expand_dims(next_obs, axis=0)

        if cfg.normalize_obs:
            next_obs = (next_obs - obs_mean) / (jnp.sqrt(obs_var) + 1e-8)

        _truncated = rollout.truncated.at[step].set(jnp.expand_dims(infos["truncation"], axis=-1))
        dones = jnp.logical_or(dones, next_dones)
        _dones = rollout.dones.at[step].set(jnp.expand_dims(dones, axis=-1))

        reward = jnp.expand_dims(reward, axis=-1)
        #_LOGGER.debug(f"Reward: {reward.shape}")
        reward = jnp.sum(reward, axis=1)

        _rewards = rollout.rewards.at[step].set(jnp.expand_dims(reward, axis=-1))

        total_rewards += reward
        ep_len += jnp.logical_not(dones)
        step += 1

        rollout = Rollout(
            obs=_obs,
            actions=_actions,
            logprobs=_logprobs,
            rewards=_rewards,
            dones=_dones,
            truncated=_truncated,
            values=_values,
        )
        return state, rollout, env_state, next_obs, dones, step, total_rewards, ep_len, key

    state, rollout, env_state, next_obs, dones, step, total_rewards, ep_len, key = nnx.while_loop(
        _cond_fn,
        _body_fn,
        (state, rollout, env_state, next_obs, dones, step, total_rewards, ep_len, key)
    )


    return PolicyEvalResult(ep_len, total_rewards), rollout
