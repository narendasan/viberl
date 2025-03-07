import jax
import jax.numpy as jnp

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    """Splits structures produced by vmap which are returned as a Structure of Arrays into a List of Structures"""
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
