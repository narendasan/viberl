from typing import Tuple

import chex
from flax import nnx
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from numpy.ma.core import exp
import optax
import logging

from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP
from viberl.algorithms.ppo._config import Config

_LOGGER = logging.getLogger(__name__)

@chex.dataclass
class PPOState:
    actor: ActorMLP
    critic: CriticMLP
    train_metrics: nnx.MultiMetric
    actor_optimizer: nnx.Optimizer
    critic_optimizer: nnx.Optimizer

def make_ppo_state(cfg: Config, env_info: Tuple[Environment, EnvParams], rngs: nnx.Rngs) -> PPOState:
    actor = ActorMLP(
        env_info[0].observation_space(env_info[1]).shape,
        env_info[0].action_space(env_info[1]).shape,
        hidden_dims=cfg.actor_hidden_dims,
        activation_fn=cfg.actor_activation_fn,
        normalize_obs=cfg.normalize_obs,
        normalize_returns=cfg.normalize_returns,
        rngs=rngs
    )
    _LOGGER.info(f"Actor network {actor}")

    critic = CriticMLP(
        env_info[0].observation_space(env_info[1]).shape,
        hidden_dims=cfg.critic_hidden_dims,
        activation_fn=cfg.critic_activation_fn,
        rngs=rngs
    )
    _LOGGER.info(f"Critic network {critic}")

    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        policy_loss=nnx.metrics.Average("policy_loss"),
        value_loss=nnx.metrics.Average("value_loss"),
        entropy=nnx.metrics.Average("entropy"),
        old_approx_kl=nnx.metrics.Average("old_approx_kl"),
        approx_kl=nnx.metrics.Average("approx_kl"),
        clipfrac=nnx.metrics.Average("clipfrac"),
        explained_var=nnx.metrics.Average("explained_var"),
        ratio_min=nnx.metrics.Average("ratio_min"),
        ratio_max=nnx.metrics.Average("ratio_max")
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

    return PPOState(
        actor=actor,
        critic=critic,
        train_metrics=train_metrics,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
