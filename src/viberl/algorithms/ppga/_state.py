from typing import Tuple

from flax import nnx

import chex
import jax.numpy as jnp

from viberl.models._actor import VectorizedActor
from viberl.models._critic import QDCritic, CriticMLP

@chex.dataclass
class VPPOState:
    actors: VectorizedActor
    qd_critic: QDCritic
    mean_critic: CriticMLP
    metrics: nnx.MultiMetric
    actor_optimizer: nnx.Optimizer
    qd_critic_optmizer: nnx.Optimizer
    mean_critic_optimizer: nnx.Optimizer
