# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from xlstm.blocks.mlstm.cell import mLSTMCell as TorchmLSTMCell

from ...components.init import bias_linspace_initializer
from ...components.ln import MultiHeadLayerNorm
from .backends import parallel_stabilized_simple, recurrent_step_stabilized_simple


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1


class mLSTMCell(nnx.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        self.rngs = rngs
        self.dtype = dtype

        # Store configuration parameters for easier access
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.dim_per_head = self.embedding_dim // self.num_heads

        # Set backend functions
        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        # Gate projections
        self.igate = nnx.Linear(
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.normal(stddev=0.1, dtype=dtype),
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.fgate = nnx.Linear(
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=bias_linspace_initializer(start=3.0, end=6.0),
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        # Output normalization
        self.outnorm = MultiHeadLayerNorm(
            num_features=config.embedding_dim,
            use_scale=True,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
        )

        # Create causal mask buffer
        # INFO: if the mask is not an instance of nnx.Param, the jit compilation
        # fails when nnx will try to flatten the mLSTMCell's pytree
        self.causal_mask = jnp.tril(
            jnp.ones((config.context_length, config.context_length), dtype=bool)
        )

    # @nnx.jit
    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass for parallel processing of the entire sequence.

        Args:
            q: Query tensor of shape (B, S, H)
            k: Key tensor of shape (B, S, H)
            v: Value tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """
        B, S, _ = q.shape  # (B, S, H)

        # Combine inputs for gate computation
        if_gate_input = jnp.concatenate([q, k, v], axis=-1)

        # Reshape for multi-head processing
        q_reshaped = q.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k_reshaped = k.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v_reshaped = v.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        # Transpose to put heads dimension before sequence dimension
        q_transposed = jnp.transpose(q_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)
        k_transposed = jnp.transpose(k_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)
        v_transposed = jnp.transpose(v_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)

        # Compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = jnp.transpose(igate_preact, (0, 2, 1))[
            ..., None
        ]  # (B, NH, S, 1)

        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = jnp.transpose(fgate_preact, (0, 2, 1))[
            ..., None
        ]  # (B, NH, S, 1)

        # Get causal mask of appropriate size
        causal_mask = self.causal_mask[:S, :S]

        # Apply mLSTM backend function
        h_state = self.backend_fn(
            queries=q_transposed,
            keys=k_transposed,
            values=v_transposed,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        # Apply normalization
        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)

        # Reshape to original dimensions
        h_state_norm = jnp.transpose(h_state_norm, (0, 2, 1, 3))  # (B, S, NH, DH)
        h_state_norm = h_state_norm.reshape(B, S, -1)  # (B, S, H)

        return h_state_norm

    def step(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mlstm_state: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Process a single step with the mLSTM cell.

        Args:
            q: Query tensor of shape (B, 1, H)
            k: Key tensor of shape (B, 1, H)
            v: Value tensor of shape (B, 1, H)
            mlstm_state: Previous state tuple (c_state, n_state, m_state) or None for initial state

        Returns:
            Tuple of output tensor (B, 1, H) and new state tuple
        """
        B, S, _ = q.shape  # (B, S, H)
        assert S == 1, (
            f"mLSTMCell.step only supports sequence length S=1, but got S={S}."
        )

        # Combine inputs for gate computation
        if_gate_input = jnp.concatenate([q, k, v], axis=-1)

        # Reshape for multi-head processing
        q_reshaped = q.reshape(B, S, self.config.num_heads, -1)  # (B, 1, NH, DH)
        k_reshaped = k.reshape(B, S, self.config.num_heads, -1)  # (B, 1, NH, DH)
        v_reshaped = v.reshape(B, S, self.config.num_heads, -1)  # (B, 1, NH, DH)

        _, _, NH, DH = q_reshaped.shape

        # Transpose to put heads dimension before sequence dimension
        q_transposed = jnp.transpose(q_reshaped, (0, 2, 1, 3))  # (B, NH, 1, DH)
        k_transposed = jnp.transpose(k_reshaped, (0, 2, 1, 3))  # (B, NH, 1, DH)
        v_transposed = jnp.transpose(v_reshaped, (0, 2, 1, 3))  # (B, NH, 1, DH)

        # Compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, 1, NH)
        igate_preact = jnp.transpose(igate_preact, (0, 2, 1))[
            ..., None
        ]  # (B, NH, 1, 1)

        fgate_preact = self.fgate(if_gate_input)  # (B, 1, NH)
        fgate_preact = jnp.transpose(fgate_preact, (0, 2, 1))[
            ..., None
        ]  # (B, NH, 1, 1)

        # Initialize state if not provided
        if mlstm_state is None:
            c_state = jnp.zeros((B, NH, DH, DH), dtype=q.dtype)
            n_state = jnp.zeros((B, NH, DH, 1), dtype=q.dtype)
            m_state = jnp.zeros((B, NH, 1, 1), dtype=q.dtype)
        else:
            c_state, n_state, m_state = mlstm_state
            # No need to convert device/dtype in JAX

        # Verify state shapes
        assert c_state.shape == (B, NH, DH, DH), (
            f"Expected c_state shape {(B, NH, DH, DH)}, but got {c_state.shape}."
        )
        assert n_state.shape == (B, NH, DH, 1), (
            f"Expected n_state shape {(B, NH, DH, 1)}, but got {n_state.shape}."
        )
        assert m_state.shape == (B, NH, 1, 1), (
            f"Expected m_state shape {(B, NH, 1, 1)}, but got {m_state.shape}."
        )

        # Apply mLSTM step function
        h_state, mlstm_state = self.backend_fn_step(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q_transposed,
            k=k_transposed,
            v=v_transposed,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )  # (B, NH, 1 DH), ((B, NH, DH, DH), (B, NH, DH, 1), (B, NH, 1, 1))

        # Apply normalization
        h_state_norm = self.outnorm(h_state)  # (B, NH, 1, DH)

        # Reshape to original dimensions
        h_state_norm = jnp.transpose(h_state_norm, (0, 2, 1, 3))  # (B, 1, NH, DH)
        h_state_norm = h_state_norm.reshape(B, S, -1)  # (B, 1, H)

        return h_state_norm, mlstm_state

    def reset_parameters(self):
        self.outnorm.reset_parameters()

        # forget gate initialization
        self.fgate.kernel = nnx.Param(jnp.zeros_like(self.fgate.kernel))

        self.fgate.bias = nnx.Param(
            jnp.linspace(3.0, 6.0, num=self.fgate.bias.value.size).reshape(
                self.fgate.bias.value.shape
            )
        )
        # input gate initialization
        self.igate.kernel = nnx.Param(jnp.zeros_like(self.igate.kernel))
        self.igate.bias = nnx.Param(jnp.zeros_like(self.igate.bias))

    def load_from_torch(self, cell: TorchmLSTMCell):
        """Load weights from a PyTorch mLSTM cell."""
        # Load weights and biases from the PyTorch model
        self.igate.kernel = nnx.Param(jnp.array(cell.igate.weight.detach().T.numpy()))
        self.igate.bias = nnx.Param(jnp.array(cell.igate.bias.detach().numpy()))

        self.fgate.kernel = nnx.Param(jnp.array(cell.fgate.weight.detach().T.numpy()))
        self.fgate.bias = nnx.Param(jnp.array(cell.fgate.bias.detach().numpy()))

        self.outnorm.load_from_torch(cell.outnorm)

        # Load the causal mask
        # self.causal_mask = nnx.Param(
        #     jnp.array(cell.get_buffer("causal_mask").detach().numpy())
        # )
