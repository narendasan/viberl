from typing import Tuple, Optional

import chex
import jax.numpy as jnp

@chex.dataclass
class Config:
    normalize_obs: bool
    normalize_returns: bool
    normalize_advantages: bool
    rollout_len: int
    num_envs: int
    num_measures: int
    value_bootstrap: bool
    gamma: float
    gae_lambda: float
    clip_coef: float
    clip_v_loss: bool
    v_clip_coef: float
    v_coef: float
    entropy_coef: float
    num_update_epochs: int
    num_minibatches: int
    target_kl: Optional[float]
