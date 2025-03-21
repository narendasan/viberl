from typing import Any, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from rejax import Algorithm
from rejax.compat import create as create_env

from rl_sandbox.utils.types import PolicyEvalResult, PolicyFn


@struct.dataclass
class EnvState:
    rng: jax.random.PRNGKey
    env_state: Any
    last_obs: chex.Array
    done: chex.Array
    reward: float = 0.0
    cum_returns: float = 0.0
    length: int = 0

def rollout_single_env(
    act: PolicyFn,  # act(obs, rng) -> action
    env,
    env_params,
    rng,
    max_steps_in_episode,
) -> Tuple[PolicyEvalResult, List[EnvState]]:

    def step_fn(state: EnvState, _) -> Tuple[EnvState, EnvState]:
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action = act(state.last_obs, rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, state.env_state, action, env_params
        )
        jax.debug.callback(lambda s, a ,r ,d: print(f"Transition: state: {s} action: {a}, reward: {r}, done: {d}"), env_state, action, reward, done)
        next_state = EnvState(
            rng=rng_step,
            env_state=env_state,
            last_obs=obs,
            done=jnp.logical_or(done, state.done),
            reward=reward.squeeze(),
            cum_returns=state.cum_returns + reward.squeeze(),
            length=state.length + 1,
        )

        return (next_state, state)

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    state = EnvState(rng=rng_eval, env_state=env_state, last_obs=obs, done=jnp.array(False))
    final_state, rollout = jax.lax.scan(step_fn, state, length=max_steps_in_episode)
    return (PolicyEvalResult(state.length, state.cum_returns), rollout)

def collect_trajectories(algo: Algorithm, agent: struct.PyTreeNode, key: chex.PRNGKey, env_name: str, env_config: Dict[str, Any], num_envs: int, max_steps_in_episode: int) -> tuple[PolicyEvalResult, List[EnvState]]:
    """Collects trajectories by running multiple environments in parallel.

    :param algo: Algorithm
        The reinforcement learning algorithm.
    :param agent: struct.PyTreeNode
        The policy of the agent as a PyTree.
    :param key: chex.PRNGKey
        Random key for environment seeds.
    :param env_name: str
        Name of the environment to create.
    :param env_config: Dict[str, Any]
        Configuration dictionary for environment creation.
    :param num_envs: int
        Number of parallel environments to run.
    :param max_steps_in_episode: int
        Maximum number of steps per episode.

    :returns:  Tuple[PolicyEvalResult, List[EnvState]]
        A tuple containing:
            - PolicyEvalResult with episode statistics
            - List of EnvState containing the trajectory information
    """
    policy = algo.make_act(agent)
    env, env_params = create_env(env_name, **env_config)
    results, trajectories = jax.vmap(lambda r: rollout_single_env(policy, env, env_params, r, max_steps_in_episode))(jax.random.split(key, num_envs))

    return results, trajectories
