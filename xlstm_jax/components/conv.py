# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Ported to JAX/Flax by Abdoul Majid O. Thiombiano
import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers


@dataclass(unsafe_hash=True, order=True)
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    # conv1d_kwargs: dict[str, tp.Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nnx.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    conv: tp.Callable[[jax.Array], jax.Array] | nnx.Conv

    def __init__(
        self,
        config: CausalConv1dConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        kernel_init=initializers.lecun_normal(),
        bias_init=initializers.zeros_init(),
    ):
        self.groups = config.feature_dim
        self.kernel_size = config.kernel_size

        if config.channel_mixing:
            self.groups = 1

        if config.kernel_size == 0:
            self.conv = jax.nn.identity
        else:
            # padding of this size assures temporal causality.
            self.pad = config.kernel_size - 1
            self.conv = nnx.Conv(
                in_features=config.feature_dim,
                out_features=config.feature_dim,
                kernel_size=(config.kernel_size,),
                padding=(self.pad,),
                feature_group_count=self.groups,
                use_bias=config.causal_conv_bias,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
            )

    @partial(jax.profiler.annotate_function, name="CausalConv1d")
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.kernel_size > 0:
            y = self.conv(x)
            # slice over the time dimension not the feature one
            y = y[:, : -self.pad, :]
            return y

        return x
