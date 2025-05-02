import jax.numpy as jnp
from flax import nnx
from xlstm import (
    FeedForwardConfig as TorchFeedForwardConfig,
)
from xlstm import (
    mLSTMBlockConfig as TorchmLSTMBlockConfig,
)
from xlstm import (
    mLSTMLayerConfig as TorchmLSTMLayerConfig,
)
from xlstm import (
    sLSTMBlockConfig as TorchsLSTMBlockConfig,
)
from xlstm import (
    sLSTMLayerConfig as TorchsLSTMLayerConfig,
)
from xlstm import (
    xLSTMLMModel as TorchxLSTMLMModel,
)
from xlstm import (
    xLSTMLMModelConfig as TorchxLSTMLMModelConfig,
)

from xlstm_jax import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMLMModel,
    xLSTMLMModelConfig,
)


def create_jax_model():
    cfg = xLSTMLMModelConfig(
        vocab_size=128,
        context_length=128,
        embedding_dim=32,
        num_blocks=2,
        slstm_at=[0],
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=4,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
    )
    rngs = nnx.Rngs(params=42)
    xlstm_stack = xLSTMLMModel(cfg, rngs=rngs, dtype=jnp.float32)

    return xlstm_stack


def create_torch_model():
    torch_cfg = TorchxLSTMLMModelConfig(
        vocab_size=128,
        context_length=128,
        embedding_dim=32,
        num_blocks=2,
        slstm_at=[0],
        mlstm_block=TorchmLSTMBlockConfig(
            mlstm=TorchmLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
            )
        ),
        slstm_block=TorchsLSTMBlockConfig(
            slstm=TorchsLSTMLayerConfig(
                backend="vanilla",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=TorchFeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
    )

    torch_xlstm_stack = TorchxLSTMLMModel(torch_cfg)

    return torch_xlstm_stack
