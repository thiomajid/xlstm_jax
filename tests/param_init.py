import jax.numpy as jnp
from flax import nnx

from .utils import create_jax_model

if __name__ == "__main__":
    model = create_jax_model()
    model.eval()
    jitted_model = nnx.jit(model)

    config = model.config
    dummy_input = jnp.ones(
        (2, config.context_length),
        dtype=jnp.uint32,
    )
    first_output = model(dummy_input)
    print("First output shape:", first_output.shape)

    jitted_output = jitted_model(dummy_input)
    print("Jitted output shape:", jitted_output.shape)
    if jnp.array_equal(first_output, jitted_output):
        print("Jitted output is the same as the first output.")
    else:
        print("Jitted output is different from the first output.")

    rngs = nnx.Rngs(42)
    model.reset_parameters(rngs)
    print("Model parameters initialized.")

    second_output = model(dummy_input)
    print("Second Output shape:", second_output.shape)

    # Check if the outputs are the same
    if jnp.array_equal(first_output, second_output):
        print("Outputs are the same after resetting parameters.")
    else:
        print("Outputs are different after resetting parameters.")
