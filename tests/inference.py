import jax.numpy as jnp

from tests.utils import create_jax_model

if __name__ == "__main__":
    model = create_jax_model()
    config = model.config

    dummy_input = jnp.ones((1, config.context_length), dtype=jnp.int32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
