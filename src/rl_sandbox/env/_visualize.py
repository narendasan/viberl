from typing import Any, Dict, List, Tuple, Callable

import os
import gymnax
from gymnax.visualize import Visualizer
import brax
from brax import envs as brax_envs
import chex
import jax
from jax._src.lax.control_flow.loops import cummax
import jax.numpy as jnp
from flax import struct
from mujoco.mjx._src.collision_convex import jp
from rejax import Algorithm
from rejax.compat import create as create_env

from viberl.utils.types import PolicyEvalResult, PolicyFn, Transition


@struct.dataclass
class EnvState:
    rng: jax.random.PRNGKey
    env_state: Any
    last_obs: chex.Array
    done: bool = False
    reward: float = 0.0
    cum_returns: float = 0.0
    length: int = 0

def rollout_single_env(
    act: PolicyFn,  # act(obs, rng) -> action
    env: Any,
    env_params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    max_steps_in_episode: int,
) -> Tuple[PolicyEvalResult, List[EnvState]]:

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    state = EnvState(rng=rng_eval, env_state=env_state, last_obs=obs)
    trajectory = []
    while state.length < max_steps_in_episode and not state.done:
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action = act(state.last_obs, rng_act)
        trajectory.append(Transition(state.env_state, action, state.reward, state.done))
        obs, env_state, reward, done, info = env.step(
            rng_step, state.env_state, action, env_params
        )
        #jax.debug.callback(lambda s, a ,r ,d: print(f"Transition: state: {s} action: {a}, reward: {r}, done: {d}"), env_state, action, reward, done)
        state = EnvState(rng=rng, env_state=env_state, last_obs=obs, reward=reward, done=done)

    print(f"Rollout reward: {state.reward}")
    return (PolicyEvalResult(state.length, state.cum_returns), trajectory)

def collect_rollouts(
    algo: Algorithm,
    agent: struct.PyTreeNode,
    key: chex.PRNGKey,
    env_name: str,
    env_config: Dict[str, Any],
    num_envs: int,
    max_steps_in_episode: int
) -> List[Tuple[PolicyEvalResult, List[EnvState]]]:
    """Collects trajectories by running multiple environments (sequentially) for the purpose of visualization.

    Parameters
    ----------
    algo : Algorithm
        The reinforcement learning algorithm
    agent : struct.PyTreeNode
        The policy of the agent as a PyTree
    key : chex.PRNGKey
        Random key for environment seeds
    env_name : str
        Name of the environment to create
    env_config : Dict[str, Any]
        Configuration dictionary for environment creation
    num_envs : int
        Number of environments to run
    max_steps_in_episode : int
        Maximum number of steps per episode

    Returns
    -------
    List[Tuple[PolicyEvalResult, List[EnvState]]]
        A list of tuples, each containing:

        * PolicyEvalResult with episode statistics
        * List of EnvState containing the trajectory information
    """
    policy = algo.make_act(agent)
    env, env_params = create_env(env_name, **env_config)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollouts = []
    for _ in range(num_envs):
        rng, rng_rollout = jax.random.split(key)
        results, trajectories = rollout_single_env(policy, env, env_params, rng_rollout, max_steps_in_episode)
        rollouts.append((results, trajectories))

    return rollouts


def render_gymnax(policy: Callable[[jax.Array, jax.Array], Tuple], config: Dict[str, Any], key: jax.Array) -> Tuple[Visualizer, float]:
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

def render_brax(policy: Callable[[jax.Array, jax.Array], Tuple], config: Dict[str, Any], key: jax.Array, steps: int=10000) -> Tuple[List, brax_envs.Env, float]:
    env = brax_envs.create(env_name=config["algorithm"]["env"].split("/")[1], backend=config["algorithm"]["env_params"]["backend"])
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
