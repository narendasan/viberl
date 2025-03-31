import jax
import jax.numpy as jnp


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
