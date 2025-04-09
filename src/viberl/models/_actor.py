from typing import Tuple, Sequence, Callable, Optional, List
import itertools

import jax
import jax.numpy as jnp
import distrax
from flax import nnx

from viberl.utils import tree_stack
from viberl.utils._pytrees import unstack_modules

# TODO: Implement obs and return normalization
class ActorMLP(nnx.Module):
    def __init__(
        self,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        *,
        hidden_dims: Sequence[int] = [128, 128],
        activation_fn: Callable = nnx.tanh,
        normalize_obs: bool=False,
        normalize_returns: bool=False,
        rngs:nnx.Rngs=nnx.Rngs(0)
    ):
        super().__init__()


        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.normalize_obs = normalize_obs
        self.normalize_returns = normalize_returns

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

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.action_mean(obs)

    def get_action(
        self,
        obs: jax.Array,
        key: jax.random.PRNGKey,
        action: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        action_mean = self.action_mean(obs)
        action_logstd = jnp.reshape(self.action_logstd.value, action_mean.shape)
        action_std = jnp.exp(action_logstd)
        dist = distrax.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample(seed=key)

        return action, dist.log_prob(action), dist.entropy()

class VectorizedActor(object):
    """
    Vectorized Actor

    Takes a singular policy (ActorMLP) and replicates it ``num_replicas`` times
    Each replica will have its own critic in QDCritic and reward function (/measure)

    This will cause the actors to diverge creating ``num_replica`` unique actors, allowing for the calculation of the Jacobian
    """

    def __init__(
        self,
        actor: ActorMLP,
        num_replicas: int,
        *,
        key: jax.random.key
    ):

        self.obs_shape = actor.obs_shape
        self.action_shape = actor.action_shape
        self.hidden_dims = actor.hidden_dims
        self.activation_fn = actor.activation_fn
        self.normalize_obs = actor.normalize_obs
        self.normalize_returns = actor.normalize_returns
        self.root_key = key
        self.num_replicas = num_replicas

        state_stack = [nnx.state(actor) for _ in range(self.num_replicas)]
        vectorized_state = tree_stack(state_stack)

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_actor_vmap(key: jax.random.key):
            return ActorMLP(
                self.obs_shape,
                self.action_shape,
                hidden_dims=self.hidden_dims,
                activation_fn=self.activation_fn,
                normalize_obs=self.normalize_obs,
                normalize_returns=self.normalize_returns,
                rngs=nnx.Rngs(key)
            )

        self._replica_keys = jnp.array([jax.random.clone(key) for _ in range(self.num_replicas)]) # Should this be split keys or cloned keys?
        self._actor_replicas = create_actor_vmap(self._replica_keys)
        nnx.update(self._actor_replicas, vectorized_state)

        @nnx.vmap(in_axes=0, out_axes=0)
        def vec_mean_action(
            replicas: ActorMLP,
            obs: jax.Array
        ):
            return replicas.action_mean(obs)

        self._vec_mean_action = vec_mean_action

        @nnx.vmap(in_axes=0, out_axes=0, axis_size=self.num_replicas)
        def vec_get_action(
            replicas: ActorMLP,
            obs: jax.Array,
            key: jax.random.key,
            action: Optional[jax.Array] = None
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            action_mean = replicas.action_mean(obs)
            action_logstd = jnp.reshape(replicas.action_logstd.value, action_mean.shape)
            action_std = jnp.exp(action_logstd)
            dist = distrax.Normal(action_mean, action_std)
            if action is None:
                print(key.shape)
                action = dist.sample(seed=key[0])
            return action, dist.log_prob(action), dist.entropy()

        self._vec_get_action = vec_get_action

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self._vec_mean_action(self._actor_replicas, obs)

    def get_action(
        self,
        obs: jax.Array,
        keys: jax.random.key,
        action: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        These should be vectorized inputs
        """
        return self._vec_get_action(self._actor_replicas, obs, keys, action)


    def unpack_actors(self) -> Sequence[ActorMLP]:
        return unstack_modules(
            ActorMLP,
            nnx.state(self._actor_replicas),
            num_slices = self.num_replicas,
            module_init_args = [(self.obs_shape, self.action_shape,) for _ in range(self.num_replicas)],
            module_init_kwargs=[{
                "hidden_dims": self.hidden_dims,
                "activation_fn": self.activation_fn,
                "normalize_obs": self.normalize_obs,
                "normalize_returns": self.normalize_returns,
                "rngs": nnx.Rngs(k)
            } for k in self._replica_keys]
        )


if __name__ == "__main__":
    actor = ActorMLP((1,4), (1,2), hidden_dims=[128, 128])
    nnx.display(actor)

    input = jnp.ones((4,))
    output = actor(input)

    print(output)
    print(actor.get_action(input, jax.random.key(0)))

    key = jax.random.key(0)
    vec_actor = VectorizedActor(actor, num_replicas=2, key=key)
    nnx.display(vec_actor._actor_replicas)


    inputs = jnp.stack([jnp.ones((4,))] * 2)
    print(inputs.shape)
    outputs = vec_actor(inputs)
    print(outputs)

    key = jax.random.key(0)
    split_keys = jnp.expand_dims(jax.random.split(key, 2), -1)
    print(inputs.shape, split_keys.shape)
    print(vec_actor.get_action(inputs, split_keys))

    [nnx.display(a) for a in vec_actor.unpack_actors()]

    get_action = jax.jit(vec_actor.get_action)
    print(get_action(inputs, split_keys))
