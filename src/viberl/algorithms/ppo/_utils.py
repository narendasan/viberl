from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def normalize(a: jax.Array, eps: float=1e-8) -> jax.Array:
    return (a - a.mean()) / (a.std() + eps)

def pg_loss(adv: jax.Array, ratio: jax.Array, *, clip_coef: float) -> jax.Array:
    l1 = -adv * ratio
    l2 = -adv * jax.lax.clamp(ratio, 1 - clip_coef, 1 + clip_coef) # Why dont we just use l2?
    return jax.lax.max(l1, l2).mean()

def v_loss(
    new_values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    *,
    clip_coef: Optional[float],
) -> jax.Array:

    if clip_coef:
        v_loss_unclipped = (new_values - returns.flatten()) ** 2
        v_clipped = old_values.flatten() + jax.lax.clamp( # ??? why use old values
            new_values - old_values.flatten(),
            -clip_coef,
            clip_coef
        )
        v_loss_clipped = (v_clipped - returns.flatten()) ** 2
        return jax.lax.max(v_loss_unclipped, v_loss_clipped)
    else:
        return ((new_values - returns.flatten()) ** 2).mean()

def calculate_discounted_sum(
    deltas: jax.Array,
    dones: jax.Array,
    discount: float,
    prev_deltas: Optional[jax.Array] = None
) -> jax.Array:

    if prev_deltas is None:
        cummulative = jnp.zeros_like(deltas[-1])
    else:
        cummulative = prev_deltas

    discounted_sum = jnp.zeros_like(deltas)

    i = len(deltas) - 1

    def _cond(carry: Tuple[int, jax.Array, jax.Array]) -> bool:
        i_, _, _ = carry
        return i_ >= 0

    def _body(carry: Tuple[int, jax.Array, jax.Array]) -> Tuple[int, jax.Array, jax.Array]:
        i_, discounted_sum_, cummulative_ = carry
        cummulative_ = deltas[i_] + discount * cummulative_ * (1.0 - dones[i_])
        discounted_sum_ = discounted_sum_.at[i_].set(cummulative_)
        return i_ - 1, discounted_sum_, cummulative_

    _, discounted_sum, _ = jax.lax.while_loop(
        _cond,
        _body,
        (i, discounted_sum, cummulative)
    )

    return discounted_sum
