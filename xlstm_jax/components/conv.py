# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

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
                rngs=rngs,
                dtype=self.dtype,
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
            y = y[:, conv_state.shape[1] :, :]

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

    def reset_parameters(self, rngs: nnx.Rngs):
        """Reset the parameters of the convolutional layer."""
        if self.config.kernel_size == 0:
            return

        # Reset kernel
        kernel_init = jax.nn.initializers.lecun_normal()
        self.conv.kernel = nnx.Param(
            kernel_init(
                rngs.params(),
                self.conv.kernel.value.shape,
                self.conv.kernel.value.dtype,
            )
        )

        # Reset bias if it exists
        if self.config.causal_conv_bias:
            bias_init = jax.nn.initializers.zeros
            self.conv.bias = nnx.Param(
                bias_init(
                    rngs.params(),
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
            jnp.array(torch_module.conv.weight.detach().numpy()).transpose(2, 1, 0)
        )

        # Load bias if it exists
        if self.config.causal_conv_bias:
            self.conv.bias = nnx.Param(
                jnp.array(torch_module.conv.bias.detach().numpy())
            )
