from collections import namedtuple
from typing import Callable, Tuple

import jax
from jax.tree_util import PyTreeDef
from rejax import Algorithm

PolicyEvalResult = namedtuple("PolicyEvalResult", 'lengths returns')
EvalCallback = Callable[[Algorithm, PyTreeDef, jax.Array, PolicyEvalResult], Tuple]
