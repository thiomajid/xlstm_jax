import jax
import jax.numpy as jnp
import torch

from .utils import create_jax_model, create_torch_model


def test_nnx_model_matches_torch_one():
    torch_xlstm_stack, xlstm_stack = create_torch_model(), create_jax_model()
    torch_x = torch.randn(4, 256, 32)
    x = jax.random.normal(jax.random.key(42), (4, 256, 32), dtype=jnp.float32)

    torch_out = torch_xlstm_stack(torch_x)
    out = xlstm_stack(x)
    assert torch_out.shape == out.shape


if __name__ == "__main__":
    test_nnx_model_matches_torch_one()
