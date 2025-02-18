from collections import namedtuple
from typing import Callable, Tuple

import chex
import jax
from flax import struct
from rejax import Algorithm

PolicyEvalResult = namedtuple("PolicyEvalResult", 'lengths returns')
Transition = namedtuple("Transition", 'state action reward done')
EvalCallback = Callable[[Algorithm, struct.PyTreeNode, jax.Array, PolicyEvalResult], Tuple]
PolicyFn = Callable[[chex.Array, chex.PRNGKey], chex.Array]
