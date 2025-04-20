import chex
from flax import nnx

from viberl.models._actor import ActorMLP
from viberl.models._critic import CriticMLP

@chex.dataclass
class PPOState:
    actor: ActorMLP
    critic: CriticMLP
    metrics: nnx.MultiMetric
    actor_optimizer: nnx.Optimizer
    critic_optimizer: nnx.Optimizer
    total_reward: chex.Array
    ep_len: chex.Array
