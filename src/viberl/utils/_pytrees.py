from typing import Any, Dict, List, Sequence, Type, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx

T = TypeVar("T")

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    """Splits structures produced by vmap which are returned as a Structure of Arrays into a List of Structures"""
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


"""
Leaving this as a note more than a utility function as the jax function does exactly what is required

You can take a PyTree like a TrainState and use `jax.flatten_util.ravel_pytree(train_state)` to flatten
all parameters to a 1D array suitable for pyribs. It also returns a function that can be used to restore the
TrainState at a later point.
"""

def unstack_modules(module_class: Type[T], stacked_module_state: nnx.State, num_slices: int, module_init_args: List[Sequence[Any]], module_init_kwargs: List[Dict[str, Any]]) -> Sequence[T]:
    module_states = [jax.tree.map (lambda x: x[i], stacked_module_state) for i in range(num_slices)]
    module_instances = [module_class(*module_init_args[i], **module_init_kwargs[i]) for i in range(num_slices)]
    [nnx.update(m, s) for m, s in zip(module_instances, module_states)]

    return module_instances

"""
A model can be vectorized using nnx.vmap, using rng or other arguments to change the initial conditions for each instance

Reusing this function and doing

state_list = tree_stack([nnx.state(model) for model in model_slices])
empty_vmap_model = create_models(jnp.full_like(consts, -1))
nnx.update(empty_vmap_model, stacked_state)

Lets you restore an unstacked set of models into a stacked one.
"""
