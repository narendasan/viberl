from collections import namedtuple
from typing import Callable, Tuple, Any

import chex
import jax
from flax import struct

PolicyEvalResult = namedtuple("PolicyEvalResult", "lengths returns")
Transition = namedtuple("Transition", "state action reward done")
EvalCallback = Callable[[struct.PyTreeNode, struct.PyTreeNode, PolicyEvalResult, struct.PyTreeNode], Tuple[Any, ...]]
PolicyFn = Callable[[chex.Array, chex.PRNGKey], chex.Array]
