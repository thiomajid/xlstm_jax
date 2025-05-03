# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from xlstm.blocks.slstm.layer import sLSTMLayer as TorchsLSTMLayer

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_initializer
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...components.ln import MultiHeadLayerNorm
from .cell import sLSTMCell, sLSTMCellConfig


@dataclass
class sLSTMLayerConfig(sLSTMCellConfig):
    embedding_dim: int = -1
    num_heads: int = 4  # this must divide the hidden size, is not yet supported by all versions in this directory
    conv1d_kernel_size: int = 4  # 0 means no convolution included
    group_norm_weight: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        self.hidden_size = self.embedding_dim
        sLSTMCellConfig.__post_init__(self)


class sLSTMLayer(nnx.Module):
    """sLSTM layer implementation in JAX/Flax."""

    config_class = sLSTMLayerConfig

    def __init__(self, config: sLSTMLayerConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype

        # Initialize convolutional layer if needed
        if self.config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                rngs=rngs,
                dtype=dtype,
                config=CausalConv1dConfig(
                    feature_dim=self.config.embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                ),
            )

        # Initialize gate projections using headwise linear layers
        gate_config = LinearHeadwiseExpandConfig(
            in_features=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            bias=False,
        )

        # We initialize all gate projections explicitly
        self.fgate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
        )

        self.igate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
        )

        self.zgate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
        )

        self.ogate = LinearHeadwiseExpand(
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
        )

        # sLSTM cell and normalization
        self.slstm_cell = sLSTMCell(self.config, rngs=rngs, dtype=dtype)
        self.group_norm = MultiHeadLayerNorm(
            num_features=self.config.embedding_dim,
            use_scale=self.config.group_norm_weight,
            rngs=rngs,
            dtype=dtype,
        )

        self.dropout = nnx.Dropout(rate=self.config.dropout, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        slstm_state: tp.Optional[jnp.ndarray] = None,
        return_last_state: bool = False,
    ) -> tp.Union[jnp.ndarray, tuple[jnp.ndarray, dict[str, tp.Any]]]:
        """Process a sequence through the sLSTM layer.

        Args:
            x: Input tensor of shape (B, S, D)
            conv_state: Initial convolution state
            slstm_state: Initial sLSTM cell state
            return_last_state: Whether to return final states

        Returns:
            Output tensor or tuple of output tensor and final states
        """
        B, S, _ = x.shape

        # Apply convolution if configured
        # if self.config.conv1d_kernel_size > 0:
        #     x_conv = self.conv1d(x)
        #     x_conv = jax.nn.swish(x_conv)  # SiLU is the same as swish
        # else:
        #     x_conv = x

        x_conv = jax.lax.cond(
            self.config.conv1d_kernel_size > 0,
            lambda _x: jax.nn.swish(self.conv1d(_x)),
            lambda _x: _x,
            operand=x,
        )

        # Apply gate projections
        f = self.fgate(x_conv)
        i = self.igate(x_conv)
        z = self.zgate(x)
        o = self.ogate(x)

        # Concatenate gate outputs into a single tensor
        gates_combined = jnp.concatenate([i, f, z, o], axis=-1)

        # Process through sLSTM cell
        y, slstm_state = self.slstm_cell(gates_combined, state=slstm_state)
        y = self.dropout(y)

        out = self.group_norm(y)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)

        if return_last_state:
            return out, {"slstm_state": slstm_state}
        else:
            return out

    def load_from_torch(self, layer: TorchsLSTMLayer):
        """Load weights from a PyTorch sLSTM layer."""

        self.conv1d.load_from_torch(layer.conv1d)
        self.fgate.load_from_torch(layer.fgate)
        self.igate.load_from_torch(layer.igate)
        self.zgate.load_from_torch(layer.zgate)
        self.ogate.load_from_torch(layer.ogate)
        self.slstm_cell.load_from_torch(layer.slstm_cell)
        self.group_norm.load_from_torch(layer.group_norm)

    def reset_parameters(self, rngs: nnx.Rngs) -> None:
        self.slstm_cell.reset_parameters(rngs)
        self.group_norm.reset_parameters(rngs)

        init_fn = small_init_initializer(dim=self.config.embedding_dim)
        self.igate.kernel = nnx.Param(
            init_fn(
                rngs.params(),
                shape=self.igate.kernel.shape,
                dtype=self.igate.kernel.dtype,
            ),
        )

        self.fgate.kernel = nnx.Param(
            init_fn(
                rngs.params(),
                shape=self.fgate.kernel.shape,
                dtype=self.fgate.kernel.dtype,
            ),
        )

        self.zgate.kernel = nnx.Param(
            init_fn(
                rngs.params(),
                shape=self.zgate.kernel.shape,
                dtype=self.zgate.kernel.dtype,
            ),
        )

        self.ogate.kernel = nnx.Param(
            init_fn(
                rngs.params(),
                shape=self.ogate.kernel.shape,
                dtype=self.ogate.kernel.dtype,
            ),
        )
