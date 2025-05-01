import jax
import jax.numpy as jnp
import numpy as np
import torch

from .utils import create_jax_model, create_torch_model


def test_nnx_model_matches_torch_one():
    torch_xlstm_stack, xlstm_stack = create_torch_model(), create_jax_model()

    # Initialize the jax model with the same parameters as the torch model
    # the goal is to have the same output values for the same input

    config = torch_xlstm_stack.config
    torch_x = torch.ones((2, config.context_length), dtype=torch.int32)
    x = jnp.array(torch_x.clone().detach().cpu().numpy())

    first_jax_out = xlstm_stack(x)
    print(f"jax out shape: {first_jax_out.shape}")

    print("Initializing the jax model with the torch model parameters")
    xlstm_stack.load_from_torch(torch_xlstm_stack)
    print("Initialized the jax model with the torch model parameters")
    out = xlstm_stack(x)

    raise SystemExit("Stop here to check the output")

    with torch.no_grad():
        torch_out = torch_xlstm_stack(torch_x)

    print(f"torch out shape: {torch_out.shape}")
    print(f"Second jax out shape: {out.shape}")

    equal = np.allclose(
        torch_out.clone().detach().cpu().numpy(),
        np.array(jax.device_get(out)),
        rtol=1e-4,
        atol=1e-4,
    )
    out

    print(f"The outputs are equal: {equal}")


if __name__ == "__main__":
    test_nnx_model_matches_torch_one()
