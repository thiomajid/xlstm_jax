# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano

from typing import Optional

import chex
import jax
import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):
    """LayerNorm but with an optional bias."""

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.ndim = ndim
        self.eps = eps
        self.rngs = rngs
        self.residual_weight = residual_weight

        self.weight = nnx.Param(jnp.zeros(ndim, dtype=dtype)) if weight else None
        self.bias = nnx.Param(jnp.zeros(ndim, dtype=dtype)) if bias else None

        self.reset_parameters()

    @property
    def weight_proxy(self) -> Optional[jnp.ndarray]:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / jnp.sqrt(var + self.eps)

        # if self.weight_proxy is not None:
        #     normed = normed * jnp.reshape(
        #         self.weight_proxy, (1,) * (x.ndim - 1) + (-1,)
        #     )

        normed = jax.lax.cond(
            self.weight_proxy is not None,
            lambda x: x * jnp.reshape(self.weight_proxy, (1,) * (x.ndim - 1) + (-1,)),
            lambda x: x,
            normed,
        )

        # if self.bias is not None:
        #     normed = normed + jnp.reshape(self.bias, (1,) * (x.ndim - 1) + (-1,))

        normed = jax.lax.cond(
            self.bias is not None,
            lambda x: x + jnp.reshape(self.bias, (1,) * (x.ndim - 1) + (-1,)),
            lambda x: x,
            normed,
        )

        return normed

    def reset_parameters(self):
        if self.weight is not None:
            if self.residual_weight:
                self.weight.value = jnp.zeros_like(self.weight)
            else:
                self.weight = jnp.ones_like(self.weight)

        if self.bias is not None:
            self.bias.value = jnp.zeros_like(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    """Applies a custom group normalization.

    Args:
        x: A 4D tensor of shape (B, NH, S, DH) where:
            B: Batch size
            NH: Number of heads / num_groups
            S: Sequence length
            DH: Dimension per head
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        chex.assert_equal(x.ndim, 4)

        B, NH, S, DH = x.shape

        # Rearrange from (B, NH, S, DH) to (B, S, NH, DH)
        gn_in_1 = jnp.transpose(x, (0, 2, 1, 3))
        # Reshape to (B * S, NH * DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)

        groups = NH
        group_size = NH * DH // groups

        # Reshape for per-group statistics
        x = gn_in_2.reshape(-1, groups, group_size)

        # Compute mean and var per group
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)

        # Normalize
        normed = (x - mean) / jnp.sqrt(var + self.eps)

        # Reshape back
        normed = normed.reshape(B * S, NH * DH)

        # Apply weight and bias if available
        # if self.weight_proxy is not None:
        #     # For group norm, we need to repeat the parameters for each group
        #     weight = jnp.repeat(self.weight_proxy[:DH], NH)
        #     x = x * weight

        normed = jax.lax.cond(
            self.weight_proxy is not None,
            lambda x: x * jnp.repeat(self.weight_proxy[:DH], NH),
            lambda x: x,
            normed,
        )

        # if self.bias is not None:
        #     # Same for bias
        #     bias = jnp.repeat(self.bias[:DH], NH)
        #     x = x + bias

        normed = jax.lax.cond(
            self.bias is not None,
            lambda x: x + jnp.repeat(self.bias[:DH], NH),
            lambda x: x,
            normed,
        )

        # Reshape back to original dimensions:
        # (B*S, NH*DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = normed.reshape(B, S, NH, DH)
        out = jnp.transpose(out, (0, 2, 1, 3))

        return out
