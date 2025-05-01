# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from flax import nnx
from xlstm import xLSTMLMModel as TorchxLSTMLMModel

from src.components.util import Identity

from .components.init import small_init_initializer
from .utils import WeightDecayOptimGroupMixin
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False


class xLSTMLMModel(WeightDecayOptimGroupMixin, nnx.Module):
    """Language model using xLSTM blocks as its backbone.

    This model combines token embeddings with an xLSTM block stack
    and a language modeling head for next token prediction.
    """

    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype

        self.xlstm_block_stack = xLSTMBlockStack(config=config, rngs=rngs, dtype=dtype)
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.embedding_dim,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
        )

        self.embedding_dropout = (
            nnx.Dropout(rate=config.dropout, rngs=rngs)
            if config.add_embedding_dropout
            else Identity()
        )

        self.lm_head = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            use_bias=False,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
        )

        # Create shared embedding parameters if using weight tying
        if config.tie_weights:
            # Create a single shared weight for both embedding and output
            initializer = small_init_initializer(dim=config.embedding_dim)
            init_fn = lambda key, shape: initializer(key, shape, dtype=dtype)  # noqa: E731

            self.shared_weight = nnx.Param(
                jnp.zeros((config.vocab_size, config.embedding_dim), dtype=dtype),
                init_fn=init_fn,
            )
        else:
            self.shared_weight = None

    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()

        # small_init_initializer is used to initialize the token embedding
        self.token_embedding.embedding = nnx.Param(
            jnp.zeros(
                (self.config.vocab_size, self.config.embedding_dim), dtype=self.dtype
            ),
            init_fn=lambda key, shape: small_init_initializer(
                dim=self.config.embedding_dim
            )(key, shape, dtype=self.dtype),
        )

        if not self.config.tie_weights:
            self.lm_head.kernel = nnx.Param(
                jnp.zeros(
                    (self.config.embedding_dim, self.config.vocab_size),
                    dtype=self.dtype,
                ),
                init_fn=lambda key, shape: small_init_initializer(
                    dim=self.config.embedding_dim
                )(key, shape, dtype=self.dtype),
            )

    # @nnx.jit
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape [B, S]

        Returns:
            Logits of shape [B, S, vocab_size]
        """

        print("Input IDs shape:", input_ids.shape)

        # Get embedding weights (either shared or dedicated)
        if self.config.tie_weights:
            emb_weight = self.shared_weight
            hidden_states = jnp.take(emb_weight, input_ids, axis=0)
        else:
            hidden_states = self.token_embedding(input_ids)

        print("Hidden states shape after embedding:", hidden_states.shape)

        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.xlstm_block_stack(hidden_states)

        # Apply language model head
        if self.config.tie_weights:
            # When weights are tied, use a functional linear layer with the shared weights
            logits = jnp.matmul(hidden_states, self.shared_weight.T)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def load_from_torch(self, torch_model: TorchxLSTMLMModel):
        """Load weights from a PyTorch xLSTM model.

        Args:
            torch_model: PyTorch xLSTM model to load weights from
        """
        # embedding layer
        self.token_embedding.embedding = nnx.Param(
            jnp.array(torch_model.token_embedding.weight.detach().numpy())
        )

        self.xlstm_block_stack.load_from_torch(torch_model.xlstm_block_stack)

        if not self.config.tie_weights:
            # lm_head layer
            self.lm_head.kernel = nnx.Param(
                jnp.array(torch_model.lm_head.weight.detach().T.numpy())
            )

            if torch_model.lm_head.bias is not None:
                self.lm_head.bias = nnx.Param(
                    jnp.array(torch_model.lm_head.bias.detach().numpy())
                )
        else:
            # shared weight layer
            self.shared_weight = nnx.Param(
                jnp.array(torch_model.lm_head.weight.detach().T.numpy())
            )

    def step(
        self,
        input_ids: jnp.ndarray,
        state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Dict[str, Any]]]:
        """Process a single step through the model.

        Args:
            input_ids: Input token indices of shape [B, 1]
            state: State dictionary from previous steps or None for initial state

        Returns:
            Tuple of logits and updated state dictionary
        """
        # Get embedding weights (either shared or dedicated)
        if self.config.tie_weights:
            emb_weight = self.shared_weight
            hidden_states = jnp.take(emb_weight, input_ids, axis=0)
        else:
            hidden_states = self.token_embedding(input_ids)

        hidden_states = self.embedding_dropout(hidden_states)

        # Process through xLSTM block stack, step by step
        hidden_states, state = self.xlstm_block_stack.step(hidden_states, state=state)

        # Apply language model head
        if self.config.tie_weights:
            # When weights are tied, use a functional linear layer with the shared weights
            logits = jnp.matmul(hidden_states, self.shared_weight.T)
        else:
            logits = self.lm_head(hidden_states)

        return logits, state

    def get_param_groups_for_optimizer(self, weight_decay=0.01):
        """Create parameter groups for optimization with weight decay control.

        This replaces the WeightDecayOptimGroupMixin from PyTorch.

        Args:
            weight_decay: Weight decay value to apply

        Returns:
            Dictionary of parameter groups with different optimization settings
        """
        # Get all parameters

        # Parameters that should have weight decay applied
        decay_params = []
        # Parameters that should not have weight decay
        no_decay_params = []

        # Sort parameters based on parameter path
        for path, param in self.named_parameters():
            # Skip parameters already in a group
            if param in decay_params or param in no_decay_params:
                continue

            # Parameters that should not have weight decay applied
            if (
                "bias" in path
                or "ln" in path  # layer norms
                or "norm" in path  # other normalizations
                or "pos_embedding" in path  # positional embeddings
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Handle embedding weight according to config
        embedding_weight = self.get_embedding_weight()

        # Remove from current groups if present
        if embedding_weight in decay_params:
            decay_params.remove(embedding_weight)
        if embedding_weight in no_decay_params:
            no_decay_params.remove(embedding_weight)

        # Add to the appropriate group based on config
        if self.config.weight_decay_on_embedding:
            decay_params.append(embedding_weight)
        else:
            no_decay_params.append(embedding_weight)

        # Return parameter groups
        return {
            "params_with_weight_decay": decay_params,
            "params_without_weight_decay": no_decay_params,
            "weight_decay": weight_decay,
        }
