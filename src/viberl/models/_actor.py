from typing import Tuple, Sequence, Callable, Optional
import itertools

import jax
import jax.numpy as jnp
import distrax
from flax import nnx

class ActorMLP(nnx.Module):
    def __init__(self,
                 obs_shape: Tuple[int],
                 action_shape: Tuple[int],
                 *,
                 hidden_dims: Sequence[int] = [128, 128],
                 activation_fn: Callable = nnx.tanh,
                 normalize_obs: bool=False,
                 normalize_returns: bool=False,
                 rngs:nnx.Rngs=nnx.Rngs(0)):
        super().__init__()

        dims = [jnp.array(obs_shape).prod().item()] + list(hidden_dims) + [jnp.array(action_shape).prod().item()]
        dim_pairs = [(dims[i], dims[i+1]) for i in range(len(dims)-1)]

        # Not sure about why the specific initialization
        kernel_init = [nnx.initializers.orthogonal(jnp.sqrt(2))] * (len(dim_pairs) - 1) + [nnx.initializers.orthogonal(0.1)]
        bias_init = [nnx.initializers.constant(0)] * len(dim_pairs)

        rngs_init = [rngs] * len(dim_pairs)

        args = [(d, {"kernel_init": k, "bias_init": b, "rngs": r}) for d, k, b, r in zip(dim_pairs, kernel_init, bias_init, rngs_init)]

        self.action_mean = nnx.Sequential(*list(itertools.chain.from_iterable([
            [nnx.Linear(*a, **k), activation_fn] for a, k in args
        ]))) #type: ignore

        self.action_logstd = nnx.Param(jnp.zeros((1, jnp.array(action_shape).prod().item())))

    def __call__(self, x):
        return self.action_mean(x)

    def get_action(self, obs, rng: jax.random.PRNGKey, action: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array, jax.Array]:
        action_mean = self.action_mean(obs)
        action_logstd = self.action_logstd.expand_as(action_mean)
        action_std = jnp.exp(action_logstd)
        dist = distrax.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample(seed=rng)

        return action, dist.log_prob(action), dist.entropy()


if __name__ == "__main__":
    actor = ActorMLP((4,), (2,), hidden_dims=[128, 128])
    nnx.display(actor)

    inputs = jnp.stack([jnp.ones((4,))] * 2)
    outputs = actor(inputs)

    print(outputs)
