import jax
import jax.numpy as jnp
import torch
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
    xLSTMBlockStack as TorchxLSTMBlockStack,
)
from xlstm import (
    xLSTMBlockStackConfig as TorchxLSTMBlockStackConfig,
)

from src import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


def __setup():
    torch_cfg = TorchxLSTMBlockStackConfig(
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
        context_length=256,
        num_blocks=2,
        embedding_dim=32,
        slstm_at=[1],
    )

    torch_xlstm_stack = TorchxLSTMBlockStack(torch_cfg)

    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
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
        context_length=256,
        num_blocks=7,
        embedding_dim=128,
        slstm_at=[1],
    )
    rngs = nnx.Rngs(params=42)
    xlstm_stack = xLSTMBlockStack(cfg, rngs=rngs, dtype=jnp.float32)

    return torch_xlstm_stack, xlstm_stack


def test_nnx_model_matches_torch_one():
    torch_xlstm_stack, xlstm_stack = __setup()
    torch_x = torch.randn(4, 256, 32)
    x = jax.random.normal(jax.random.PRNGKey(42), (4, 256, 32), dtype=jnp.float32)

    torch_out = torch_xlstm_stack(torch_x)
    out = xlstm_stack(x)
    assert torch_out.shape == out.shape


if __name__ == "__main__":
    test_nnx_model_matches_torch_one()
