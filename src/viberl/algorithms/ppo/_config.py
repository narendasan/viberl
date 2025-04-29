from typing import Callable, Optional, Sequence

import chex
import flax

@chex.dataclass(unsafe_hash=True, frozen=True)
class _TrainingSettingConfigSubset:
    normalize_obs: bool
    normalize_returns: bool
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


@chex.dataclass(unsafe_hash=True, frozen=True)
class Config:
    normalize_obs: bool
    normalize_returns: bool
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
    actor_activation_fn: Callable[[chex.Array], chex.Array]
    critic_activation_fn: Callable[[chex.Array], chex.Array]
    actor_lr: float
    critic_lr: float
    actor_max_grad_norm: float
    critic_max_grad_norm: float
    total_timesteps: int
    weight_decay: float
    eval_frequency: int

    def training_config_subset(self) -> _TrainingSettingConfigSubset:
        return _TrainingSettingConfigSubset(
            normalize_obs=self.normalize_obs,
            normalize_returns=self.normalize_returns,
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

def make_config(**kwargs) -> Config:
    if "actor_activation_fn" in kwargs and isinstance(kwargs["actor_activation_fn"], str):
        kwargs["actor_activation_fn"] = getattr(flax.nnx, kwargs["actor_activation_fn"])
    if "critic_activation_fn" in kwargs and isinstance(kwargs["critic_activation_fn"], str):
        kwargs["critic_activation_fn"] = getattr(flax.nnx, kwargs["critic_activation_fn"])
    return Config(**kwargs)
