from typing import Any, Dict, List, Tuple

import chex
import jax
from flax import struct
from rejax import Algorithm
from rejax.compat import create as create_env

from rl_sandbox.utils.types import PolicyEvalResult, PolicyFn, Transition


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

    rollouts = []
    for _ in range(num_envs):
        rng, rng_rollout = jax.random.split(key)
        results, trajectories = rollout_single_env(policy, env, env_params, rng_rollout, max_steps_in_episode)
        rollouts.append((results, trajectories))

    return rollouts
