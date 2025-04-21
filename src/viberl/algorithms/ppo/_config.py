from typing import Optional, Sequence, Callable

import chex

@chex.dataclass
class Config:
    normalize_obs: bool
    normalize_returns: bool
    normalize_advantages: bool
    rollout_len: int
    num_envs: int
    num_measures: int
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
