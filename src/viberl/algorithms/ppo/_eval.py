import logging
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment

from viberl.algorithms.ppo._config import Config
from viberl.algorithms.ppo._rollout import Rollout, make_empty_rollout
from viberl.algorithms.ppo._state import PPOState

_LOGGER = logging.getLogger(__name__)

def eval(
    state: PPOState,
    cfg: Config,
    env_info: Tuple[Environment, EnvParams],
    key: jax.random.key,
    *
    collect_values: bool,
    eval_callback: Optional[Callable[[PPOState, Config, Rollout, bool], None]]
) -> Tuple[jax.Array, Rollout]:

    total_rewards = jnp.zeros((cfg.num_envs,))
    ep_len = jnp.zeros((cfg.num_envs,))
    dones = jnp.zeros((cfg.num_envs,))

    env, env_params = env_info
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    key, reset_key = jax.random.split(key, 2)
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, cfg.num_envs), env_params)

    if len(next_obs.shape) == 1:
        next_obs = jnp.expand_dims(next_obs, axis=0)

    if cfg.normalize_obs:
        next_obs = state.actor.normalize_obs(next_obs)

    rollout = make_empty_rollout(
        cfg.rollout_len,
        cfg.num_envs,
        env.observation_space(env_params).shape,
        env.action_space(env_params).shape,
    )

    _LOGGER.debug(f"Rollout: {rollout.shapes}")

    while not jnp.all(dones):
        key, action_key, env_step_key = jax.random.split(key, 3)

        _obs = rollout.obs.at[step].set(next_obs)
        action, logprob, _ = state.actor.get_action(next_obs, key=action_key)
        _LOGGER.debug(f"Action: {action.shape}, logprob {logprob.shape}")

        _actions = rollout.actions.at[step].set(action)
        _logprobs = rollout.logprobs.at[step].set(logprob)

        if collect_values:
            value = state.critic(next_obs)
            _LOGGER.debug(f"Value: {value.shape}")
            _values = rollout.values.at[step].set(value)
        else:
            _values = rollout.values

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

        _LOGGER.info(f"Eval: Total Reward: {total_rewards} Episode Len: {ep_len}")

    if eval_callback is not None:
        eval_callback(state, cfg, rollout, collect_values)

    return total_rewards, rollout
