from typing import Callable, Sequence
import itertools

import jax
from flax import nnx
from rejax.networks import VNetwork

class QDCriticNetwork(nnx.Module):
    def __init__(self, hidden_layer_sizes: Sequence[int], activation: Callable, measure_dim: int, rngs: nnx.Rngs):
        super().__init__()

        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._measure_dim = measure_dim

        self._critics = [VNetwork(hidden_layer_sizes, activation)] * (measure_dim + 1) # type: ignore

    def __call__(self, x: jax.Array) -> jax.Array:
