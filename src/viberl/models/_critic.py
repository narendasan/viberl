from typing import Tuple, Sequence, Callable, Optional, List
import itertools

from distrax._src.utils.transformations import lu
from flax.nnx.statelib import K
from gymnax.environments import EnvState
import jax
import jax.numpy as jnp
import distrax
from flax import nnx
from numpy import int_
from viberl.utils import tree_stack

class CriticMLP(nnx.Module):
    def __init__(self,
                 obs_shape: Tuple[int],
                 *,
                 hidden_dims: Sequence[int] = [256, 265],
                 activation_fn: Callable = nnx.tanh,
                 rngs:nnx.Rngs=nnx.Rngs(0)):
        super().__init__()

        dims = [jnp.array(obs_shape).prod().item()] + list(hidden_dims) + [1]
        dim_pairs = [(dims[i], dims[i+1]) for i in range(len(dims)-1)]

        # Not sure about why the specific initialization
        kernel_init = [nnx.initializers.orthogonal(jnp.sqrt(2))] * (len(dim_pairs) - 1) + [nnx.initializers.orthogonal(0.1)]
        bias_init = [nnx.initializers.constant(0)] * len(dim_pairs)
        rngs_init = [rngs] * len(dim_pairs)

        args = [(d, {"kernel_init": k, "bias_init": b, "rngs": r}) for d, k, b, r in zip(dim_pairs, kernel_init, bias_init, rngs_init)]

        self.critic_ = nnx.Sequential(list(itertools.chain.from_iterable([
            [nnx.Linear(*a, **k), activation_fn] for a, k in args
        ]))) #type: ignore

    def __call__(self, x):
        return self.critic_(x)

class QDCritic(object):
    def __init__(self,
                 obs_shape: Tuple[int],
                 *,
                 hidden_dims: Sequence[int] = [256, 265],
                 activation_fn: Callable = nnx.tanh,
                 num_critics: Optional[int] = None,
                 critic_list: Optional[List[nnx.Module]] = None,
                 key: jax.random.key):
        super().__init__()

        assert (num_critics is not None) ^ (critic_list is not None), "Exactly one of num_critics or critic_list must be provided"

        if critic_list is None:
            assert num_critics is not None, "num_critics must be provided since critic_list is None"
            key_splits = jax.random.split(key, num_critics)
            critic_state_list = [nnx.state(CriticMLP(obs_shape, hidden_dims=hidden_dims, activation_fn=activation_fn, rngs=nnx.Rngs(k))) for k in key_splits]
        else:
            num_critics = len(critic_list)
            key_splits = jax.random.split(key, num_critics)
            critic_state_list = [nnx.state(c) for c in critic_list]

        stacked_state = tree_stack(critic_state_list)

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_vmap_critic(key: jax.random.key):
            return CriticMLP(obs_shape, hidden_dims=hidden_dims, activation_fn=activation_fn, rngs=nnx.Rngs(key))

        self.critic_ = create_vmap_critic(key_splits)

    def get_value_at(self, obs: jax.Array, idx: int):
        val = self.critic_(obs)
        return val[idx]

    def get_all_values(self, obs: jax.Array):
        val = self.critic_(obs)
        return val

if __name__ == "__main__":
    critic = CriticMLP((4,), hidden_dims=[256, 256])
    nnx.display(critic)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    qd_critic_1 = QDCritic((4,), hidden_dims=[256, 256], num_critics=2, key=k1)
    nnx.display(qd_critic_1.critic_)

    qd_critic_2 = QDCritic((4,), hidden_dims=[256, 256], critic_list=[critic, critic], key=k2)
    nnx.display(qd_critic_2.critic_)
