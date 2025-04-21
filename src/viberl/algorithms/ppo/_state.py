import chex
from flax import nnx
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from numpy.ma.core import exp
import optax

from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP
from viberl.algorithms.ppo._config import Config

@chex.dataclass
class PPOState:
    actor: ActorMLP
    critic: CriticMLP
    train_metrics: nnx.MultiMetric
    actor_optimizer: nnx.Optimizer
    critic_optimizer: nnx.Optimizer
    total_reward: chex.Array
    ep_len: chex.Array

def make_ppo_state(cfg: Config, env: Environment, rngs: nnx.Rngs) -> PPOState:
    actor = ActorMLP(
        env.observation_space(),
        env.action_space(),
        hidden_dims=cfg.actor_hidden_dims,
        activation_fn=cfg.actor_activation_fn,
        normalize_obs=cfg.normalize_obs,
        normalize_returns=cfg.normalize_returns,
        rngs=rngs
    )
    critic = CriticMLP(
        env.observation_space(),
        hidden_dims=cfg.critic_hidden_dims,
        activation_fn=cfg.critic_activation_fn,
        rngs=rngs
    )
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(),
        policy_loss=nnx.metrics.Average(),
        value_loss=nnx.metrics.Average(),
        entropy=nnx.metrics.Average(),
        old_approx_kl=nnx.metrics.Average(),
        approx_kl=nnx.metrics.Average(),
        clipfrac=nnx.metrics.Average(),
        explained_var=nnx.metrics.Average(),
        ratio_min=nnx.metrics.Average(),
        ratio_max=nnx.metrics.Average()
    )

    actor_optimizer = nnx.Optimizer(
        actor, optax.chain(
            optax.clip_by_global_norm(max_norm=cfg.actor_max_grad_norm),
            optax.adam(learning_rate=cfg.actor_lr)
        )
    )
    critic_optimizer = nnx.Optimizer(
        critic, optax.chain(
            optax.clip_by_global_norm(max_norm=cfg.critic_max_grad_norm),
            optax.adam(learning_rate=cfg.critic_lr)
        )
    )
    total_reward = jnp.zeros((cfg.num_envs,))
    ep_len = jnp.zeros((cfg.num_envs,))

    return PPOState(
        actor,
        critic,
        train_metrics,
        actor_optimizer,
        critic_optimizer,
        total_reward,
        ep_len)
