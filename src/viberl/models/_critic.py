from typing import Tuple, Sequence, Callable, Optional, List
import itertools

import jax
import jax.numpy as jnp
import distrax
from flax import nnx
from numpy.lib.index_tricks import r_

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

class QDCriticMLP(nnx.Module):
    def __init__(self,
                 obs_shape: Tuple[int],
                 *,
                 hidden_dims: Sequence[int] = [256, 265],
                 activation_fn: Callable = nnx.tanh,
                 num_critics: Optional[int] = None,
                 critic_list: Optional[List[nnx.Module]] = None,
                 rngs:nnx.Rngs=nnx.Rngs(0)):
        super().__init__()

        assert (num_critics is not None) ^ (critic_list is not None), "Exactly one of num_critics or critic_list must be provided"

        if num_critics is not None:
            ensemble_rngs = nnx.split_rngs(rngs, num_critics)
            self.critic_list = [CriticMLP(obs_shape, hidden_dims=hidden_dims, activation_fn=activation_fn, rngs=r) for r in ensemble_rngs]
        else:
            self.critic_list = critic_list


if __name__ == "__main__":
    actor = CriticMLP((4,), hidden_dims=[256, 256])
    nnx.display(actor)
