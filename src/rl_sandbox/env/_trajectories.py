from typing import Any, Dict, Tuple

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
    done: bool = False
    reward: float = 0.0
    cum_reward: float = 0.0
    length: int = 0

@struct.dataclass
class Trajectory:
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array


def rollout_single_env(
    act: PolicyFn,  # act(obs, rng) -> action
    env,
    env_params,
    rng,
    max_steps_in_episode,
) -> Tuple[PolicyEvalResult, Trajectory]:

    def step_fn(prev: Tuple[EnvState, Trajectory]) -> Tuple[EnvState, Trajectory]:
        state, trajectory = prev
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action = act(state.last_obs, rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, state.env_state, action, env_params
        )
        next_state = EnvState(
            rng=rng_step,
            env_state=env_state,
            last_obs=obs,
            done=done,
            reward = reward,
            cum_reward=state.reward + reward.squeeze(),
            length=state.length + 1,
        )
        new_trajectory = Trajectory(
            trajectory.obs.at[state.length + 1].set(next_state.last_obs),
            trajectory.actions.at[state.length].set(action),
            trajectory.rewards.at[state.length + 1].set(next_state.reward),
        )

        return (next_state, trajectory)

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    state = EnvState(rng=rng_eval, env_state=env_state, last_obs=obs)
    trajectory = Trajectory(
        obs=jnp.empty((max_steps_in_episode, *state.last_obs.shape)),
        actions=jnp.empty((max_steps_in_episode, *env.action_space(env_params).shape)),
        rewards=jnp.empty((max_steps_in_episode,))
    )
    state, trajectory = jax.lax.while_loop(
        lambda s: jnp.logical_and(
            s[0].length < max_steps_in_episode, jnp.logical_not(s[0].done)
        ),
        step_fn,
        (state, trajectory),
    )
    return (PolicyEvalResult(state.length, state.cum_reward), trajectory)

def collect_trajectories(algo: Algorithm, agent: struct.PyTreeNode, key: chex.PRNGKey, env_name: str, env_config: Dict[str, Any]) -> (None):
    policy = algo.make_act(agent)
    env, env_params = create_env(env_name, **env_config)
    jax.jit(jax.vmap(lambda r: rollout_single_env(policy, env, env_params, r, 1)))(jax.random.split(key, 10))

    return None
