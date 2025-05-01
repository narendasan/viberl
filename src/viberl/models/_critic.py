import itertools
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from viberl.utils import tree_stack
from viberl.utils._pytrees import unstack_modules


class CriticMLP(nnx.Module):
    def __init__(self,
                 obs_shape: Tuple[int],
                 *,
                 hidden_dims: Sequence[int] = [256, 265],
                 activation_fn: Callable | str = nnx.tanh,
                 rngs:nnx.Rngs=nnx.Rngs(0)):
        super().__init__()

        self.id = jax.random.key_data(rngs())

        if isinstance(activation_fn, str):
            activation_fn = getattr(nnx, activation_fn)

        dims = [jnp.array(obs_shape).prod().item()] + list(hidden_dims)
        dim_pairs = [(dims[i], dims[i+1]) for i in range(len(dims)-1)]

        # Not sure about why the specific initialization
        kernel_init = [nnx.initializers.orthogonal(jnp.sqrt(2))] * (len(dim_pairs) - 1)
        bias_init = [nnx.initializers.constant(0)] * len(dim_pairs)
        rngs_init = [rngs] * len(dim_pairs)

        args = [(d, {"kernel_init": k, "bias_init": b, "rngs": r}) for d, k, b, r in zip(dim_pairs, kernel_init, bias_init, rngs_init)]

        self.core_ = nnx.Sequential(*list(itertools.chain.from_iterable([
            [nnx.Linear(*a, **k), activation_fn] for a, k in args
        ]))) #type: ignore

        self.critic_ = nnx.Linear(
            dims[-1], 1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(0),
            rngs=rngs
        )

    def __call__(self, x):
        z = self.core_(x)
        return self.critic_(z)

class QDCritic(object):
    def __init__(self,
                 obs_shape: Tuple[int],
                 *,
                 hidden_dims: Sequence[int] = [256, 265],
                 activation_fn: Callable | str = nnx.tanh,
                 num_critics: Optional[int] = None,
                 critic_list: Optional[Sequence[nnx.Module]] = None,
                 key: jax.random.key):
        super().__init__()

        assert (num_critics is not None) ^ (critic_list is not None), "Exactly one of num_critics or critic_list must be provided"

        # Need to store so that when unpacking, the networks can be reconstructed
        self.obs_shape = obs_shape
        self.hidden_dims = hidden_dims
        if isinstance(activation_fn, str):
            activation_fn = getattr(nnx, activation_fn)
        self.activation_fn = activation_fn


        if critic_list is None:
            assert num_critics is not None, "num_critics must be provided since critic_list is None"
            key_splits = jax.random.split(key, num_critics)
            critic_state_list = [nnx.state(CriticMLP(obs_shape, hidden_dims=self.hidden_dims, activation_fn=self.activation_fn, rngs=nnx.Rngs(k))) for k in key_splits]
        else:
            num_critics = len(critic_list)
            key_splits = jax.random.split(key, num_critics)
            critic_state_list = [nnx.state(c) for c in critic_list]

        self.num_critics = num_critics
        self.key_splits = key_splits

        stacked_state = tree_stack(critic_state_list)

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_vmap_critic(key: jax.random.key):
            return CriticMLP(obs_shape, hidden_dims=hidden_dims, activation_fn=activation_fn, rngs=nnx.Rngs(key))

        @nnx.vmap(in_axes=(0,0), out_axes=0)
        def run_models(models, inputs):
            return models(inputs)

        self._run_models = run_models

        self.critic_ = create_vmap_critic(key_splits)
        nnx.update(self.critic_, stacked_state)

    def get_all_values(self, obs: jax.Array):
        return self._run_models(self.critic_, obs)

    def get_value_at(self, obs: jax.Array, idx: int):
        val = self.get_all_values(obs)
        return val[idx]

    def unpack_critics(self):
        return unstack_modules(
            CriticMLP,
            nnx.state(self.critic_),
            num_slices=self.num_critics,
            module_init_args=[(self.obs_shape,) for _ in range(self.num_critics)],
            module_init_kwargs=[{
                "hidden_dims": self.hidden_dims,
                "activation_fn": self.activation_fn,
                "rngs": nnx.Rngs(k) # Does this change the seed?
            } for k in self.key_splits]
        )

if __name__ == "__main__":
    critic = CriticMLP((4,), hidden_dims=[256, 256])
    nnx.display(critic)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    qd_critic_1 = QDCritic((4,), hidden_dims=[256, 256], num_critics=2, key=k1)
    nnx.display(qd_critic_1.critic_)

    qd_critic_2 = QDCritic((4,), hidden_dims=[256, 256], critic_list=[critic, critic], key=k2)
    nnx.display(qd_critic_2.critic_)

    critic_list = qd_critic_2.unpack_critics()
    [nnx.display(critic) for critic in critic_list]

    qd_critic_3 = QDCritic((4,), hidden_dims=[256, 256], critic_list=critic_list, key=k2)
    nnx.display(qd_critic_3.critic_)

    inputs = jnp.stack([jnp.ones((4,))] * 2)

    out1 = qd_critic_1.get_all_values(inputs)
    out2 = qd_critic_2.get_all_values(inputs)
    out3 = qd_critic_3.get_all_values(inputs)

    print(out1, out2, out3)

    assert jnp.allclose(out2, out3), "Vectorized outputs should be identical"
