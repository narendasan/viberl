from typing import Callable, Optional, Sequence, Self, Dict, Any

import copy
import jax
from flax import nnx
from flax import struct

@struct.dataclass
class _TrainingConfig:
    normalize_obs: bool
    normalize_rewards: bool
    normalize_advantages: bool
    rollout_len: int
    num_envs: int
    v_bootstrap: bool
    gamma: float
    gae_lambda: float
    surrogate_clip_coef: float
    clip_v_loss: bool
    v_clip_coef: float
    v_coef: float
    entropy_coef: float
    num_update_epochs: int
    num_minibatches: int
    target_kl: Optional[float]
    total_timesteps: int
    weight_decay: float

@struct.dataclass
class _EvalConfig:
    normalize_obs: bool
    normalize_rewards: bool
    normalize_advantages: bool
    rollout_len: int
    eval_episodes: int

@struct.dataclass
class Config:
    normalize_obs: bool
    normalize_rewards: bool
    normalize_advantages: bool
    rollout_len: int
    num_envs: int
    v_bootstrap: bool
    gamma: float
    gae_lambda: float
    surrogate_clip_coef: float
    clip_v_loss: bool
    v_clip_coef: float
    v_coef: float
    entropy_coef: float
    num_update_epochs: int
    num_minibatches: int
    target_kl: Optional[float]
    actor_hidden_dims: Sequence[int]
    critic_hidden_dims: Sequence[int]
    actor_activation_fn: Callable[[jax.Array], jax.Array]
    critic_activation_fn: Callable[[jax.Array], jax.Array]
    lr: float
    use_lr_schedule: bool
    max_grad_norm: float
    total_timesteps: int
    weight_decay: float
    eval_frequency: int
    num_measures: int
    eval_episodes: int

    def training_config_subset(self) -> _TrainingConfig:
        return _TrainingConfig(
            normalize_obs=self.normalize_obs,
            normalize_rewards=self.normalize_rewards,
            normalize_advantages=self.normalize_advantages,
            rollout_len=self.rollout_len,
            num_envs=self.num_envs,
            v_bootstrap=self.v_bootstrap,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            surrogate_clip_coef=self.surrogate_clip_coef,
            clip_v_loss=self.clip_v_loss,
            v_clip_coef=self.v_clip_coef,
            v_coef=self.v_coef,
            entropy_coef=self.entropy_coef,
            num_update_epochs=self.num_update_epochs,
            num_minibatches=self.num_minibatches,
            target_kl=self.target_kl,
            total_timesteps=self.total_timesteps,
            weight_decay=self.weight_decay
        )


    def eval_config_subset(self) -> _EvalConfig:
        return _EvalConfig(
            normalize_obs=self.normalize_obs,
            normalize_rewards=self.normalize_rewards,
            normalize_advantages=self.normalize_advantages,
            rollout_len=self.rollout_len,
            eval_episodes=self.eval_episodes,
        )

    @classmethod
    def from_dict(cls: Self, kwargs: Dict[str, Any]) -> Self:
        kwargs = copy.deepcopy(kwargs)
        if "actor_activation_fn" in kwargs and isinstance(kwargs["actor_activation_fn"], str):
            kwargs["actor_activation_fn"] = getattr(nnx, kwargs["actor_activation_fn"])
        if "critic_activation_fn" in kwargs and isinstance(kwargs["critic_activation_fn"], str):
            kwargs["critic_activation_fn"] = getattr(nnx, kwargs["critic_activation_fn"])
        return cls(**kwargs)
