# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import functools
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from xlstm.components.conv import CausalConv1d as TorchCausalConv1d


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


def conv1d_step(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    conv1d_weight: jnp.ndarray,
    conv1d_bias: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    B: batch size
    S: sequence length
    D: feature dimension
    KS: kernel size
    Args:
        x (jnp.ndarray): (B, S, D)
        conv_state (jnp.ndarray): (B, KS, D)
        conv1d_weight (jnp.ndarray): (KS, D)
    """
    # Ensure the batch size and feature dimension match
    chex.assert_equal(x.shape[0], conv_state.shape[0])

    # Ensure the feature dimension matches
    chex.assert_equal(x.shape[2], conv_state.shape[2])

    # Ensure the sequence length is 1
    chex.assert_equal(x.shape[1], 1)

    # Create a new state by shifting and updating, instead of in-place modification
    new_conv_state = jnp.roll(conv_state, shift=-1, axis=1).at[:, -1:, :].set(x)

    # Calculate the convolution output
    y = jnp.sum(new_conv_state * conv1d_weight, axis=1, keepdims=True)
    y = jax.lax.cond(
        conv1d_bias is not None,
        lambda: y + conv1d_bias,
        lambda: y,
    )

    return y, new_conv_state


class CausalConv1d(nnx.Module):
    config_class = CausalConv1dConfig
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

    def __init__(
        self,
        config: CausalConv1dConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config
        self.rngs = rngs
        self.dtype = dtype
        self.groups = self.config.feature_dim
        if self.config.channel_mixing:
            self.groups = 1

        if self.config.kernel_size == 0:
            # No convolution needed
            self.conv = None
        else:
            self.pad = (
                self.config.kernel_size - 1
            )  # padding of this size assures temporal causality.

            self.conv = nnx.Conv(
                in_features=self.config.feature_dim,
                out_features=self.config.feature_dim,
                kernel_size=self.config.kernel_size,
                padding=self.pad,
                feature_group_count=self.groups,
                kernel_init=jax.nn.initializers.lecun_normal(),
                use_bias=self.config.causal_conv_bias,
                bias_init=jax.nn.initializers.zeros,
                rngs=self.rngs,
                dtype=self.dtype,
            )

    @functools.partial(
        nnx.jit,
        static_argnames=("return_last_state",),
    )
    def __call__(
        self,
        x: jnp.ndarray,
        conv_state: Optional[jnp.ndarray] = None,
        return_last_state: bool = False,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        if conv_state is not None:
            x = jnp.concatenate([conv_state, x], axis=1)

        if self.config.kernel_size == 0:
            if return_last_state:
                return x, None
            return x

        y = x  # With nnx.Conv the feature dimension is the last one so no need to transpose
        y = self.conv(y)  # (B, T+pad, F) tensor

        # Handle the case when conv_state is provided
        if conv_state is not None:
            y = y[:, conv_state.shape[2] :, :]

        # output = jnp.transpose(y[:, :, : -self.pad], (0, 2, 1))
        # no need to transpose since the feature dimension is already the last one
        # and no transposition was done pre convolution
        output = y[:, : -self.pad, :]  # the time dimension is the second one

        if return_last_state:
            last_state = jax.lax.cond(
                self.pad > 0,
                lambda: x[:, -self.pad :],
                lambda: None,
            )

            return output, last_state
        else:
            return output

    def step(
        self,
        x: jnp.ndarray,
        conv_state: Optional[Tuple[jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
        """Process a single step with the convolutional layer."""
        if self.config.kernel_size == 0:
            return x, conv_state

        B, S, D = x.shape

        # Initialize state if not provided
        if conv_state is None:
            conv_state = (jnp.zeros((B, self.config.kernel_size, D), dtype=self.dtype),)

        # Extract weights in the format needed for conv1d_step
        if self.groups == 1:
            # For channel mixing, we need to reshape the weight
            conv1d_weight = jnp.transpose(self.weight[:, 0, :], (1, 0))
        else:
            # For depthwise, weights are already in the right shape
            conv1d_weight = jnp.transpose(self.weight[:, 0, :], (1, 0))

        # Process a single step
        y, new_conv_state = conv1d_step(
            x,
            conv_state[0],
            conv1d_weight,
            conv1d_bias=self.bias
            if self.config.causal_conv_bias and self.bias is not None
            else None,
        )

        return y, (new_conv_state,)

    def reset_parameters(self):
        """Reset the parameters of the convolutional layer."""
        if self.config.kernel_size == 0:
            return

        # Reset kernel
        kernel_init = jax.nn.initializers.lecun_normal()
        key = self.rngs.params()
        self.conv.kernel = nnx.Param(
            kernel_init(
                key,
                self.conv.kernel.value.shape,
                self.conv.kernel.value.dtype,
            )
        )

        # Reset bias if it exists
        if self.config.causal_conv_bias:
            bias_init = jax.nn.initializers.zeros
            self.conv.bias = nnx.Param(
                bias_init(
                    key,
                    self.conv.bias.value.shape,
                    self.conv.bias.value.dtype,
                )
            )

    def _create_weight_decay_optim_groups(
        self,
    ) -> Tuple[Tuple[nnx.Param, ...], Tuple[nnx.Param, ...]]:
        if self.config.kernel_size == 0:
            return (), ()
        else:
            weight_decay = (self.conv.kernel,)
            no_weight_decay = ()
            if self.config.causal_conv_bias and self.bias is not None:
                no_weight_decay = (self.conv.bias,)
            return weight_decay, no_weight_decay

    def load_from_torch(
        self,
        torch_module: TorchCausalConv1d,
    ):
        """Load weights from a PyTorch module."""
        if self.config.kernel_size == 0:
            return

        # Load kernel
        self.conv.kernel = nnx.Param(
            jnp.array(torch_module.conv.weight.data.numpy()).transpose(2, 0, 1)
        )

        # Load bias if it exists
        if self.config.causal_conv_bias:
            self.conv.bias = nnx.Param(jnp.array(torch_module.bias.data.numpy()))
