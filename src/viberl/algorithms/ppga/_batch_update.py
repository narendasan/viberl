import logging
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from viberl.algorithms.ppga._config import _TrainingConfig
from viberl.algorithms.ppga._rollout import Rollout
from viberl.algorithms.ppga._state import TrainState

train_fn = Callable[[TrainState, _TrainingConfig, Rollout, jax.Array, jax.Array, jax.Array], Tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array,  jax.Array, jax.Array, jax.Array]]
returns_fn = Callable[[TrainState, _TrainingConfig, Rollout], Tuple[jax.Array, jax.Array]]

_LOGGER = logging.getLogger(__name__)

def batch_update(
    state: TrainState,
    cfg: _TrainingConfig,
    rollout: Rollout,
    *,
    calculate_returns_fn: returns_fn,
    train_step_fn: train_fn
) -> Tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    @nnx.jit(static_argnums=(1,))
    def _batch_update(
        state: TrainState,
        cfg: _TrainingConfig,
        rollout: Rollout
    ) -> Tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        advantages, returns = jax.lax.stop_gradient(calculate_returns_fn(state, cfg, rollout))


        batch_size = rollout.obs.shape[0]
        minibatch_size = batch_size // cfg.num_minibatches
        #_LOGGER.debug(f"Batch size: {batch_size}, minibatch size: {minibatch_size}")

        batch_idxs = jnp.arange(batch_size)
        mb_idxs = batch_idxs.reshape(-1, minibatch_size)

        pg_loss = v_loss = entropy_loss = ratio = approx_kl = clipfracs = jnp.empty((cfg.num_update_epochs, cfg.num_minibatches))
        ratio = jnp.empty((cfg.num_update_epochs, cfg.num_minibatches, minibatch_size))

        # TODO: JAXIFY
        def _update_epoch(
            e: int,
            carry: Tuple[TrainState, _TrainingConfig, Rollout, jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]
        ) -> Tuple[TrainState, _TrainingConfig, Rollout, jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array,jax.Array, jax.Array]]:
            state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio) = carry
            state, _, _pg_loss, _v_loss, _entropy_loss, _approx_kl, _clipfracs, _ratio = train_step_fn(state, cfg, rollout, advantages, returns, mb_idxs)
            pg_loss = pg_loss.at[e].set(_pg_loss)
            v_loss = v_loss.at[e].set(_v_loss)
            entropy_loss = entropy_loss.at[e].set(_entropy_loss)
            ratio = ratio.at[e].set(_ratio)
            approx_kl = approx_kl.at[e].set(_approx_kl)
            clipfracs = clipfracs.at[e].set(_clipfracs)
            return state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)

        (state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)) = nnx.fori_loop(
            0,
            cfg.num_update_epochs,
            _update_epoch,
            (state, cfg, rollout, advantages, returns, mb_idxs, (pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, ratio)),
        )


        ratio_min = ratio[-1][-1].min()
        ratio_max = ratio[-1][-1].max()

        return state, pg_loss[-1][-1], v_loss[-1][-1], entropy_loss[-1][-1], approx_kl[-1][-1], clipfracs[-1], ratio[-1][-1]
    return _batch_update(state, cfg, rollout)
