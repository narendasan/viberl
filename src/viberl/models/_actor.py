import itertools
from typing import Callable, Optional, Sequence, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from viberl.utils import tree_stack
from viberl.utils._pytrees import unstack_modules


class ActorMLP(nnx.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        *,
        hidden_dims: Sequence[int] = [128, 128],
        activation_fn: Callable | str = nnx.tanh,
        normalize_epsilon: float = 1e-8,
        rngs:nnx.Rngs=nnx.Rngs(0)
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_dims = hidden_dims
        self.id = jax.random.key_data(rngs())

        if isinstance(activation_fn, str):
            activation_fn = getattr(nnx, activation_fn)
        self.activation_fn = activation_fn

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

        self.obs_mean = nnx.Variable(jnp.ones(self.obs_shape), )
        self.obs_var = nnx.Variable(jnp.ones(self.obs_shape))
        self.obs_count = nnx.Variable(jnp.ones((1,)))
        self.rewards_mean = nnx.Variable(jnp.ones(1,))
        self.rewards_var = nnx.Variable(jnp.ones((1,)))
        self.rewards_count = nnx.Variable(jnp.ones((1,)))
        self.normalize_eps = normalize_epsilon

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.action_mean(obs)


    def get_action(
        self,
        obs: jax.Array,
        *,
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        action_mean = self.action_mean(obs)
        action_logstd = jnp.broadcast_to(self.action_logstd.value, action_mean.shape)
        action_std = jnp.exp(action_logstd)
        dist = distrax.Normal(action_mean, action_std)
        action = dist.sample(seed=key)

        return action, jnp.expand_dims(dist.log_prob(action).sum(axis=1), axis=-1), dist.entropy()

    def get_action_log_probs(
        self,
        obs: jax.Array,
        *,
        action: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        action_mean = self.action_mean(obs)
        action_logstd = jnp.broadcast_to(self.action_logstd.value, action_mean.shape)
        action_std = jnp.exp(action_logstd)
        dist = distrax.Normal(action_mean, action_std)

        return jnp.expand_dims(dist.log_prob(action).sum(axis=1), axis=-1), dist.entropy()

    @staticmethod
    @jax.jit
    def _update_running_stats(
        batch: jax.Array,
        mean: jax.Array,
        var: jax.Array,
        count: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def normalize_obs(self, batch: jax.Array) -> jax.Array:
        self.obs_mean.value, self.obs_var.value, self.obs_count.value = ActorMLP._update_running_stats(
            batch, self.obs_mean.value, self.obs_var.value, self.obs_count.value
        )
        return (batch - self.obs_mean) / jnp.sqrt(self.obs_var + self.normalize_eps)

    def normalize_rewards(self, batch: jax.Array) -> jax.Array:
        self.rewards_mean.value, self.rewards_var.value, self.rewards_count.value = ActorMLP._update_running_stats(
            batch, self.rewards_mean.value, self.rewards_var.value, self.rewards_count.value
        )
        return jax.lax.clamp(-5.0, (batch - self.rewards_mean) / jnp.sqrt(self.rewards_var + self.normalize_eps), 5.0)


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
        normalize_epsilon: float=1e-8,
        *,
        key: jax.Array
    ):

        self.obs_shape = actor.obs_shape
        self.action_shape = actor.action_shape
        self.hidden_dims = actor.hidden_dims
        self.activation_fn = actor.activation_fn
        self.root_key = key
        self.num_replicas = num_replicas

        self.obs_mean = nnx.Variable(jnp.ones(self.obs_shape), )
        self.obs_var = nnx.Variable(jnp.ones(self.obs_shape))
        self.obs_count = nnx.Variable(jnp.ones((1,)))
        self.rewards_mean = nnx.Variable(jnp.ones(1,))
        self.rewards_var = nnx.Variable(jnp.ones((1,)))
        self.rewards_count = nnx.Variable(jnp.ones((1,)))
        self.normalize_eps = normalize_epsilon

        state_stack = [nnx.state(actor) for _ in range(self.num_replicas)]
        vectorized_state = tree_stack(state_stack)

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_actor_vmap(key: jax.Array):
            return ActorMLP(
                self.obs_shape,
                self.action_shape,
                hidden_dims=self.hidden_dims,
                activation_fn=self.activation_fn,
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
            key: jax.Array,
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            action_mean = replicas.action_mean(obs)
            action_logstd = jnp.broadcast_to(replicas.action_logstd.value, action_mean.shape)
            action_std = jnp.exp(action_logstd)
            dist = distrax.Normal(action_mean, action_std)
            action = dist.sample(seed=key[0])
            return action, jnp.expand_dims(dist.log_prob(action).sum(axis=1), axis=-1), dist.entropy()

        self._vec_get_action = vec_get_action

        @nnx.vmap(in_axes=0, out_axes=0, axis_size=self.num_replicas)
        def vec_get_action_log_probs(
            replicas: ActorMLP,
            obs: jax.Array,
            action: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            action_mean = replicas.action_mean(obs)
            action_logstd = jnp.broadcast_to(replicas.action_logstd.value, action_mean.shape)
            action_std = jnp.exp(action_logstd)
            dist = distrax.Normal(action_mean, action_std)
            return jnp.expand_dims(dist.log_prob(action).sum(axis=1), axis=-1), dist.entropy()

        self._vec_get_action_log_probs = vec_get_action_log_probs

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self._vec_mean_action(self._actor_replicas, obs)

    def get_action(
        self,
        obs: jax.Array,
        keys: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        These should be vectorized inputs
        """
        return self._vec_get_action(self._actor_replicas, obs, keys)

    def get_action_log_probs(
        self,
        obs: jax.Array,
        actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        These should be vectorized inputs
        """
        return self._vec_get_action_log_probs(self._actor_replicas, obs, actions)

    def normalize_obs(self, batch: jax.Array) -> jax.Array:
        self.obs_mean.value, self.obs_var.value, self.obs_count.value = ActorMLP._update_running_stats(
            batch, self.obs_mean.value, self.obs_var.value, self.obs_count.value
        )
        return (batch - self.obs_mean) / jnp.sqrt(self.obs_var + self.normalize_eps)

    def normalize_rewards(self, batch: jax.Array) -> jax.Array:
        self.rewards_mean.value, self.rewards_var.value, self.rewards_count.value = ActorMLP._update_running_stats(
            batch, self.rewards_mean.value, self.rewards_var.value, self.rewards_count.value
        )
        return jax.lax.clamp(-5.0, (batch - self.rewards_mean) / jnp.sqrt(self.rewards_var + self.normalize_eps), 5.0)

    def unpack_actors(self) -> Sequence[ActorMLP]:
        return unstack_modules(
            ActorMLP,
            nnx.state(self._actor_replicas),
            num_slices = self.num_replicas,
            module_init_args = [(self.obs_shape, self.action_shape,) for _ in range(self.num_replicas)],
            module_init_kwargs=[{
                "hidden_dims": self.hidden_dims,
                "activation_fn": self.activation_fn,
                "rngs": nnx.Rngs(k)
            } for k in self._replica_keys]
        )


if __name__ == "__main__":
    actor0 = ActorMLP((1,4), (1,2), hidden_dims=[128, 128])
    nnx.display(actor0)

    input = jnp.ones((1, 4,))
    output = actor0(input)

    print(output)
    print(actor0.get_action(input, key=jax.random.key(0)))

    actor_state = nnx.state(actor0, nnx.Param)
    flattened_state, restore_fn = jax.flatten_util.ravel_pytree(actor_state)
    print(flattened_state)

    actor = ActorMLP((1,4), (1,2), hidden_dims=[128, 128])
    reconstructed_state = restore_fn(flattened_state)
    nnx.update(actor, reconstructed_state)
    nnx.display(actor)

    input = jnp.ones((4,))
    output1 = actor(input)

    print(output1)
    assert (output == output1).all()

    key = jax.random.key(0)
    vec_actor = VectorizedActor(actor, num_replicas=2, key=key)
    nnx.display(vec_actor._actor_replicas)


    inputs = jnp.stack([jnp.ones((1, 4,))] * 2)
    print(inputs.shape)
    outputs = vec_actor(inputs)
    print(outputs)

    key = jax.random.key(0)
    split_keys = jnp.expand_dims(jax.random.split(key, 2), -1)
    print(inputs.shape, split_keys.shape)
    print(vec_actor.get_action(inputs, keys=split_keys))

    key = jax.random.key(0)
    split_keys = jnp.expand_dims(jax.random.split(key, 2), -1)
    print(inputs.shape, split_keys.shape)
    a, lp1, e1 = vec_actor.get_action(inputs, keys=split_keys)
    lp2, e2 = vec_actor.get_action_log_probs(inputs, a)
    assert all([(lp1 == lp2).all(), (e1 == e2).all()])

    [nnx.display(a) for a in vec_actor.unpack_actors()]


    jit_action = nnx.jit(vec_actor._vec_get_action)
    jit_action(vec_actor._actor_replicas, inputs, split_keys)
    print(vec_actor._vec_get_action(vec_actor._actor_replicas, inputs, split_keys))
