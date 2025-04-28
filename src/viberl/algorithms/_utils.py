import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

_LOGGER = logging.getLogger(__name__)

def normalize(a: jax.Array, eps: float=1e-8) -> jax.Array:
    return (a - a.mean()) / (a.std() + eps)

def policy_grad_loss(adv: jax.Array, ratio: jax.Array, *, clip_coef: float) -> jax.Array:
    l1 = -adv * ratio
    l2 = -adv * jax.lax.clamp(1 - clip_coef, ratio, 1 + clip_coef) # Why dont we just use l2?
    return jax.lax.max(l1, l2).mean()

def value_loss(
    new_values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    *,
    clip_coef: Optional[float],
) -> jax.Array:
    _LOGGER.debug(f"new_values: {new_values.shape}, old_values: {old_values.shape}, returns: {returns.shape}")

    if clip_coef is not None:
        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = old_values + jax.lax.clamp( # ??? why use old values
            -clip_coef,
            new_values - old_values,
            clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = jax.lax.max(v_loss_unclipped, v_loss_clipped)
        return v_loss_max.mean()
    else:
        return ((new_values - returns) ** 2).mean()

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
