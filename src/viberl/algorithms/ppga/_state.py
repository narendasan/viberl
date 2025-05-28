from sched import scheduler
import logging
from typing import Tuple, Dict, Any, Self, Callable

from jax import numpy as jnp
import jax
import optax
from flax import nnx
from flax import struct
from gymnax.environments import EnvParams
from gymnax.environments.environment import Environment

from viberl.algorithms.ppga._config import Config
from viberl.models._actor import ActorMLP, VectorizedActor
from viberl.models._critic import QDCritic, CriticMLP

_LOGGER = logging.getLogger(__name__)

@struct.dataclass(frozen=False)
class State:
    actor: ActorMLP
    qd_critic: QDCritic
    mean_critic: CriticMLP
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
        key, actor_key, qd_critic_key, mean_critic_key = jax.random.split(key, 3)
        actor = ActorMLP(
            obs_shape,
            action_shape,
            hidden_dims=cfg.actor_hidden_dims,
            activation_fn=cfg.actor_activation_fn,
        )
        _LOGGER.info(f"Actor network {actor}")

        qd_critic = QDCritic(obs_shape, hidden_dims=cfg.critic_hidden_dims, activation_fn=cfg.critic_activation_fn, num_critics=cfg.num_measures + 1, key=qd_critic_key)
        _LOGGER.info(f"QD Critic network {qd_critic}")


        mean_critic = CriticMLP(
            obs_shape,
            hidden_dims=cfg.critic_hidden_dims,
            activation_fn=cfg.critic_activation_fn,
            rngs=flax.nnx.Rngs(mean_critic_key)
        )

        _LOGGER.info(f"Mean Critic network {mean_critic}")

        return cls(
            actor=actor,
            qd_critic=qd_critic,
            mean_critic=mean_critic,
        )

    @classmethod
    def from_env(cls, cfg: Config, *, env_info: Tuple[Environment, EnvParams], key: jax.Array) -> Self:
        return cls.from_shape(cfg, obs_shape=env_info[0].observation_space(env_info[1]).shape, action_shape=env_info[0].action_space(env_info[1]).shape, key=key)

    def policy_fn(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
        def _policy(obs: jax.Array, key: jax.Array) -> jax.Array:
            key, action_key, = jax.random.split(key, 2)
            action, _, _ = self.actor_critic.actor.get_action(obs, key=action_key)
            return jnp.squeeze(action, axis=0)

        return _policy

class VecActorQDCritic(nnx.Module):
    def __init__(self, actor: VectorizedActor, critic: QDCritic):
        super().__init__()
        self.actor = actor
        self.critic = critic

class TrainState(nnx.Optimizer):
    def __init__(self, model, tx):
        super().__init__(model, tx)
        self.global_step = 0

        self.train_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            policy_loss=nnx.metrics.Average("policy_loss"),
            value_loss=nnx.metrics.Average("value_loss"),
            entropy=nnx.metrics.Average("entropy"),
            approx_kl=nnx.metrics.Average("approx_kl"),
            clipfrac=nnx.metrics.Average("clipfrac"),
            explained_var=nnx.metrics.Average("explained_var"),
            ratio_min=nnx.metrics.Average("ratio_min"),
            ratio_max=nnx.metrics.Average("ratio_max"),
            lr=nnx.metrics.Average("lr")
        )

        self.rollout_metrics = nnx.MultiMetric(
            reward=nnx.metrics.Average("train_reward"),
        )


        self.eval_metrics = nnx.MultiMetric(
            reward=nnx.metrics.Average("reward"),
            ep_len=nnx.metrics.Average("ep_len"),
        )

    def update(self, *, grads):
       super().update(grads)


def  make_train_state(model: nnx.Module, cfg: Config) -> TrainState:
    num_train_per_eval = cfg.eval_frequency // (cfg.rollout_len * cfg.num_envs)
    assert num_train_per_eval > 0, "Eval Frequency must be >= Rollout Len * Num Envs"
    num_updates = cfg.total_timesteps //  (cfg.rollout_len * cfg.num_envs)
    num_grad_updates = num_updates * cfg.num_update_epochs * cfg.num_minibatches
    lr_schedule = optax.linear_schedule(cfg.lr, cfg.lr * 0.0001, num_grad_updates) if cfg.use_lr_schedule else cfg.lr

    @optax.inject_hyperparams
    def optimizer_chain(lr_schedule):
        return optax.chain(
            optax.clip_by_global_norm(max_norm=cfg.max_grad_norm),
            optax.adam(learning_rate=lr_schedule)
        )

    train_state = TrainState(
        model, optimizer_chain(lr_schedule)
    )

    return train_state
