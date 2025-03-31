import os
from typing import Any, Callable, Dict, List, Tuple

import brax
import chex
import gymnax
import jax
import jax.numpy as jnp
from brax import envs as brax_envs
from flax import struct
from gymnax.visualize import Visualizer
from jax._src.lax.control_flow.loops import cummax
from mujoco.mjx._src.collision_convex import jp
from rejax import Algorithm
from rejax.compat import create as create_env

from viberl.utils.types import PolicyEvalResult, PolicyFn, Transition


def render_gymnax(
    policy: Callable[[jax.Array, jax.Array], Tuple],
    config: Dict[str, Any],
    key: jax.Array,
) -> Tuple[Visualizer, float]:
    env, env_params = gymnax.make(config["algorithm"]["env"].split("/")[1])

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    root_key, reset_key = jax.random.split(key)

    rollout = []
    obs, state = jit_env_reset(reset_key, env_params)
    cum_reward = 0
    done = False
    info = None

    while True:
        rollout.append(state)
        root_key, step_key, act_key = jax.random.split(root_key, 3)
        act = policy(obs, act_key)
        obs, state, reward, done, info = jit_env_step(step_key, state, act, env_params)
        cum_reward += reward
        if done:
            break

    vis = Visualizer(env, env_params, rollout, jnp.array([cum_reward]))
    return vis, cum_reward


def render_brax(
    policy: Callable[[jax.Array, jax.Array], Tuple],
    config: Dict[str, Any],
    key: jax.Array,
    steps: int = 10000,
) -> Tuple[List, brax_envs.Env, float]:
    env = brax_envs.create(
        env_name=config["algorithm"]["env"].split("/")[1],
        backend=config["algorithm"]["env_params"]["backend"],
    )
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollout = []
    state = jit_env_reset(rng=key)
    reward = 0
    for _ in range(steps):
        reward += state.reward
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(key)
        act = policy(state.obs, act_rng)
        state = jit_env_step(state, act)

    return rollout, env, reward
