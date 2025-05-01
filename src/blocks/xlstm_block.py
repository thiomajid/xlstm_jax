# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from flax import nnx
from xlstm.blocks.xlstm_block import xLSTMBlock as TorchxLSTMBlock

from src.components.ln import LayerNorm

from ..components.feedforward import FeedForwardConfig, create_feedforward
from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from .slstm.layer import sLSTMLayer, sLSTMLayerConfig


@dataclass
class xLSTMBlockConfig:
    mlstm: Optional[mLSTMLayerConfig] = None
    slstm: Optional[sLSTMLayerConfig] = None
    feedforward: Optional[FeedForwardConfig] = None

    # we initialize these with None to catch the case where they are not set
    _num_blocks: Optional[int] = None
    _block_idx: Optional[int] = None

    def __post_init__(self):
        assert self.mlstm is not None or self.slstm is not None, (
            "Either mlstm or slstm must be provided"
        )
        assert self.mlstm is None or self.slstm is None, (
            "Only one of mlstm or slstm can be provided"
        )

        embedding_dim = (
            self.mlstm.embedding_dim
            if self.mlstm is not None
            else self.slstm.embedding_dim
        )

        if self.mlstm:
            self.mlstm._num_blocks = self._num_blocks
            self.mlstm._block_idx = self._block_idx

        if self.slstm:
            self.slstm._num_blocks = self._num_blocks
            self.slstm._block_idx = self._block_idx

        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()


class xLSTMBlock(nnx.Module):
    """An xLSTM block can be either an sLSTM Block or an mLSTM Block.

    It contains the pre-LayerNorms and the skip connections.
    """

    config_class = xLSTMBlockConfig

    def __init__(
        self,
        config: xLSTMBlockConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ) -> None:
        """Initialize an xLSTM block.

        Args:
            config: Configuration for the xLSTM block
        """
        self.config = config
        self.dtype = dtype
        self.rngs = rngs

        embedding_dim = (
            self.config.mlstm.embedding_dim
            if self.config.mlstm is not None
            else self.config.slstm.embedding_dim
        )

        self.xlstm_norm = LayerNorm(
            num_features=embedding_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
        )

        if self.config.mlstm is not None:
            self.xlstm = mLSTMLayer(config=self.config.mlstm, rngs=rngs, dtype=dtype)
        elif self.config.slstm is not None:
            self.xlstm = sLSTMLayer(config=self.config.slstm, rngs=rngs, dtype=dtype)
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(
                num_features=self.config.feedforward.embedding_dim,
                use_bias=False,
                rngs=rngs,
                dtype=dtype,
            )

            self.ffn = create_feedforward(
                config=self.config.feedforward,
                rngs=rngs,
                dtype=dtype,
            )
        else:
            self.ffn_norm = None
            self.ffn = None

    def __call__(self, x: jnp.ndarray):
        """Process a full sequence through the xLSTM block.

        Args:
            x: Input tensor of shape (B, S, D)

        Returns:
            Output tensor of shape (B, S, D)
        """
        x = x + self.xlstm(self.xlstm_norm(x))
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x))

        # x = lax.cond(
        #     self.ffn is not None,
        #     lambda x: x + self.ffn(self.ffn_norm(x)),
        #     lambda x: x,
        #     x,
        # )

        return x

    def step(
        self,
        x: jnp.ndarray,
        states: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Process a single step through the xLSTM block.

        Args:
            x: Input tensor of shape (B, 1, D)
            states: Dictionary of previous states or None for initial states

        Returns:
            Tuple of output tensor and updated states dictionary
        """
        # Initialize empty states dictionary if None provided
        if states is None:
            states = {}

        # Extract xlstm states from the states dictionary
        xlstm_states = {}
        for k in ["mlstm_state", "conv_state"]:
            if k in states:
                xlstm_states[k] = states[k]

        # Apply xlstm layer in step mode
        xlstm_output, new_states = self.xlstm.step(
            self.xlstm_norm(x), **xlstm_states, **kwargs
        )

        # Add residual connection
        x = x + xlstm_output

        # Apply feedforward network if present
        if self.ffn is not None:
            ffn_input = self.ffn_norm(x)
            ffn_output = self.ffn(ffn_input)
            x = x + ffn_output

        return x, new_states

    def load_from_torch(self, torch_block: TorchxLSTMBlock) -> None:
        """Load parameters from a PyTorch xLSTM block.

        Args:
            torch_block: PyTorch xLSTM block to load parameters from
        """

        self.xlstm_norm.load_from_torch(torch_block.xlstm_norm)
        self.xlstm.load_from_torch(torch_block.xlstm)
        if self.ffn is not None:
            self.ffn.load_from_torch(torch_block.ffn)
            self.ffn_norm.load_from_torch(torch_block.ffn_norm)

    def reset_parameters(self) -> None:
        """Reset parameters of the xLSTM block."""
        self.xlstm_norm.reset_parameters()
        self.xlstm.reset_parameters()
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()
