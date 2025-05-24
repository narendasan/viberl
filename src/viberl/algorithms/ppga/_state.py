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

from viberl.algorithms.ppga._config import Config
from viberl.models._actor import ActorMLP, VectorizedActor
from viberl.models._critic import QDCritic

_LOGGER = logging.getLogger(__name__)

class VecActorQDCritic(nnx.Module):
    def __init__(self, actor: VectorizedActor, critic: QDCritic):
        super().__init__()
        self.actor = actor
        self.critic = critic

@struct.dataclass(frozen=False)
class State:
    actor_critic: VecActorQDCritic
    train_metrics: nnx.MultiMetric
    eval_metrics: nnx.MultiMetric
    rollout_metrics: nnx.MultiMetric
    optimizer: nnx.Optimizer
    global_step: int = 0

    def _generate_checkpoint(self):
        actor_state = nnx.state(self.actor_critic.actor)
        critic_state = nnx.state(self.actor_critic.critic)
        return {
            "global_step": self.global_step,
            "actor": actor_state,
            "critic": critic_state,
            #"actor_optimizer": self.actor_optimizer.opt_state,
            #"critic_optimizer": self.critic_optimizer.opt_state
        }

    def load_ckpt_dict(self, ckpt: Dict[str, Any]) -> None:
        nnx.update(self.actor_critic.actor, ckpt["actor"])

        _LOGGER.info(f"Actor network {self.actor_critic.actor}")

        nnx.update(self.actor_critic.critic, ckpt["critic"])

        _LOGGER.info(f"Critic network {self.actor_critic.critic}")

        #self.actor_optimizer.opt_state = tree_util.tree_unflatten(
        #    tree_util.tree_structure(self.actor_optimizer.opt_state), tree_util.tree_leaves(ckpt["actor_optimizer"])
        #)

        #self.critic_optimizer.opt_state = tree_util.tree_unflatten(
        #    tree_util.tree_structure(self.critic_optimizer.opt_state), tree_util.tree_leaves(ckpt["critic_optimizer"])
        #)
        self.global_step = ckpt["global_step"]

    @classmethod
    def from_shape(cls, cfg: Config, *, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], key: jax.Array) -> Self:
        key, actor_key, critic_key = jax.random.split(key, 3)
        actor_base = ActorMLP(
            obs_shape,
            action_shape,
            hidden_dims=cfg.actor_hidden_dims,
            activation_fn=cfg.actor_activation_fn,
            normalize_obs=cfg.normalize_obs,
            normalize_rewards=cfg.normalize_rewards,
        )

        actor = VectorizedActor(actor_base, num_replicas=cfg.num_measures + 1, key=actor_key)

        _LOGGER.info(f"Actor network {actor}")

        # critic_base = CriticMLP(
        #     obs_shape,
        #     hidden_dims=cfg.critic_hidden_dims,
        #     activation_fn=cfg.critic_activation_fn,
        #     rngs=rngs
        # )

        critic = QDCritic(obs_shape, hidden_dims=cfg.critic_hidden_dims, activation_fn=cfg.critic_activation_fn, num_critics=cfg.num_measures + 1, key=critic_key)
        _LOGGER.info(f"Critic network {critic}")

        actor_critic = VecActorQDCritic(actor=actor, critic=critic)

        train_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            policy_loss=nnx.metrics.Average("policy_loss"),
            value_loss=nnx.metrics.Average("value_loss"),
            entropy=nnx.metrics.Average("entropy"),
            approx_kl=nnx.metrics.Average("approx_kl"),
            clipfrac=nnx.metrics.Average("clipfrac"),
            explained_var=nnx.metrics.Average("explained_var"),
            ratio_min=nnx.metrics.Average("ratio_min"),
            ratio_max=nnx.metrics.Average("ratio_max"),
        )

        rollout_metrics = nnx.MultiMetric(
            reward=nnx.metrics.Average("train_reward"),
        )


        eval_metrics = nnx.MultiMetric(
            reward=nnx.metrics.Average("reward"),
            ep_len=nnx.metrics.Average("ep_len"),
        )

        optimizer = nnx.Optimizer(
            actor_critic, optax.chain(
                optax.clip_by_global_norm(max_norm=cfg.max_grad_norm),
                optax.adam(learning_rate=cfg.lr)
            )
        )

        return cls(
            actor_critic=actor_critic,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            rollout_metrics=rollout_metrics,
            optimizer=optimizer,
        )

    @classmethod
    def from_env(cls, cfg: Config, *, env_info: Tuple[Environment, EnvParams], rngs: nnx.Rngs) -> Self:
        return cls.from_shape(cfg, obs_shape=env_info[0].observation_space(env_info[1]).shape, action_shape=env_info[0].action_space(env_info[1]).shape, rngs=rngs)

    def policy_fn(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
        def _policy(obs: jax.Array, key: jax.Array) -> jax.Array:
            key, action_key, = jax.random.split(key, 2)
            action, _, _ = self.actor_critic.actor.get_action(obs, key=action_key)
            return jnp.squeeze(action, axis=0)

        return _policy
