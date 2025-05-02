import logging
from typing import Tuple, Dict, Any, Self, Callable, overload

from jax import numpy as jnp
import jax
import optax
from flax import nnx
from flax import struct
from jax import tree_util
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment

from viberl.algorithms.ppo._config import Config
from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP

_LOGGER = logging.getLogger(__name__)

@struct.dataclass(frozen=False)
class State:
    actor: ActorMLP
    critic: CriticMLP
    train_metrics: nnx.MultiMetric
    actor_optimizer: nnx.Optimizer
    critic_optimizer: nnx.Optimizer
    global_step: int = 0

    def _generate_checkpoint(self):
        actor_state = nnx.state(self.actor)
        critic_state = nnx.state(self.critic)
        return {
            "global_step": self.global_step,
            "actor": actor_state,
            "critic": critic_state,
            #"actor_optimizer": self.actor_optimizer.opt_state,
            #"critic_optimizer": self.critic_optimizer.opt_state
        }

    def load_ckpt_dict(self, ckpt: Dict[str, Any]) -> None:
        nnx.update(self.actor, ckpt["actor"])

        _LOGGER.info(f"Actor network {self.actor}")

        nnx.update(self.critic, ckpt["critic"])

        _LOGGER.info(f"Critic network {self.critic}")

        #self.actor_optimizer.opt_state = tree_util.tree_unflatten(
        #    tree_util.tree_structure(self.actor_optimizer.opt_state), tree_util.tree_leaves(ckpt["actor_optimizer"])
        #)

        #self.critic_optimizer.opt_state = tree_util.tree_unflatten(
        #    tree_util.tree_structure(self.critic_optimizer.opt_state), tree_util.tree_leaves(ckpt["critic_optimizer"])
        #)
        self.global_step = ckpt["global_step"]

    @classmethod
    def from_shape(cls, cfg: Config, *, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], rngs: nnx.Rngs) -> Self:
        actor = ActorMLP(
            obs_shape,
            action_shape,
            hidden_dims=cfg.actor_hidden_dims,
            activation_fn=cfg.actor_activation_fn,
            normalize_obs=cfg.normalize_obs,
            normalize_returns=cfg.normalize_returns,
            rngs=rngs
        )
        _LOGGER.info(f"Actor network {actor}")

        critic = CriticMLP(
            obs_shape,
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

        return cls(
            actor=actor,
            critic=critic,
            train_metrics=train_metrics,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )

    @classmethod
    def from_env(cls: Self, cfg: Config, *, env_info: Tuple[Environment, EnvParams], rngs: nnx.Rngs) -> Self:
        return cls.from_shape(cfg, obs_shape=env_info[0].observation_space(env_info[1]).shape, action_shape=env_info[0].action_space(env_info[1]).shape, rngs=rngs)

    def policy_fn(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
        def _policy(obs: jax.Array, key: jax.Array) -> jax.Array:
            key, action_key, = jax.random.split(key, 2)
            action, _, _ = self.actor.get_action(obs, key=action_key)
            return jnp.squeeze(action, axis=0)

        return _policy
