# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano

import typing as tp

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.nnx.nn.normalization import _compute_stats, _normalize
from flax.typing import Axes, Dtype, Initializer
from xlstm.components.ln import LayerNorm as TorchLayerNorm


class LayerNorm(nnx.Module):
    """LayerNorm but with a residual_scale proxy"""

    def __init__(
        self,
        num_features: int = -1,
        epsilon: float = 1e-6,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = initializers.zeros_init(),
        scale_init: Initializer = initializers.ones_init(),
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: tp.Optional[str] = None,
        axis_index_groups: tp.Any = None,
        use_fast_variance: bool = True,
        residual_scale: bool = True,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        feature_shape = (num_features,)

        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = None

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = None

        self.num_features = num_features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance
        self.residual_scale = residual_scale

    @property
    def scale_proxy(self) -> tp.Optional[jnp.ndarray]:
        if self.scale is None:
            return None
        if self.residual_scale:
            return 1.0 + self.scale
        else:
            return self.scale

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        mask: tp.Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Applies layer normalization on the input.

        Args:
        x: the inputs

        Returns:
        Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            x,
            mean,
            var,
            self.scale_proxy if self.scale else None,
            self.bias.value if self.bias else None,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.epsilon,
        )

    def reset_parameters(self):
        if self.scale is not None:
            key = self.rngs.params()
            if self.residual_scale:
                self.scale = nnx.Param(
                    jax.nn.initializers.zeros(
                        key, self.scale.shape, dtype=self.scale.dtype
                    )
                )
            else:
                self.scale = nnx.Param(
                    self.scale(key, self.scale.shape, dtype=self.scale.dtype)
                )

        if self.bias is not None:
            self.bias = nnx.Param(
                self.bias_init(
                    self.rngs.params(), self.bias.shape, dtype=self.bias.dtype
                )
            )

    def load_from_torch(self, torch_ln: TorchLayerNorm) -> None:
        """Load weights from a PyTorch LayerNorm module."""
        if self.scale is not None:
            self.scale.value = nnx.Param(jnp.array(torch_ln.weight.data.numpy()))

        if self.bias is not None:
            self.bias.value = nnx.Param(jnp.array(torch_ln.bias.data.numpy()))


class MultiHeadLayerNorm(nnx.Module):
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
        # if self.scale_proxy is not None:
        #     # For group norm, we need to repeat the parameters for each group
        #     weight = jnp.repeat(self.scale_proxy[:DH], NH)
        #     x = x * weight

        normed = jax.lax.cond(
            self.scale_proxy is not None,
            lambda x: x * jnp.repeat(self.scale_proxy[:DH], NH),
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
