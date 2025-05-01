from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from tqdm import tqdm

from .utils import create_jax_model, create_torch_model


def test_nnx_model_matches_torch_one():
    torch_xlstm_stack, xlstm_stack = create_torch_model(), create_jax_model()

    xlstm_stack.eval()
    torch_xlstm_stack = torch_xlstm_stack.eval()

    # Initialize the jax model with the same parameters as the torch model
    # the goal is to have the same output values for the same input

    config = torch_xlstm_stack.config
    torch_x = torch.randint(1, config.vocab_size, (10, 1), dtype=torch.int32)
    x = jnp.array(torch_x.clone().detach().cpu().numpy(), dtype=jnp.int32)

    print("Initializing the jax model with the torch model parameters")
    xlstm_stack.load_from_torch(torch_xlstm_stack)
    print("Initialized the jax model with the torch model parameters")

    MAX_NEW_TOKEN = 10
    # Jit the model once
    jitted_stack = nnx.jit(xlstm_stack)
    # Run a warm-up call to trigger compilation before timing
    print("Compiling JAX model...")
    # Determine expected shapes for warm-up if model needs fixed size
    # For now, assume it handles variable length up to final length
    initial_len = x.shape[1]
    final_len = initial_len + MAX_NEW_TOKEN
    batch_size = x.shape[0]
    # Warm-up with the final shape might be safer if model expects fixed size
    warmup_x = jnp.zeros((batch_size, final_len), dtype=jnp.int32)
    warmup_x = warmup_x.at[:, :initial_len].set(x)
    _ = jitted_stack(warmup_x)  # Warm-up with max length input
    print("Compilation complete.")

    # --- Setup for JAX Scan ---
    # Pre-allocate the full sequence array
    full_x_init = jnp.zeros((batch_size, final_len), dtype=jnp.int32)
    full_x_init = full_x_init.at[:, :initial_len].set(x)
    # Initial carry: (full sequence array, index to write next token)
    initial_carry_val = (full_x_init, initial_len)
    vocab_size = config.vocab_size  # Get vocab size from config

    # Define the function for a single generation step for scan
    def generation_step(carry, _):
        current_full_x, current_index = carry

        # --- Input Preparation ---
        # Option 1: Pass the whole buffer (if model handles masking/padding)
        # input_sequence = current_full_x
        # Option 2: Slice the relevant part dynamically (more robust)
        # Assuming model needs sequence up to current_index
        # input_sequence = jax.lax.dynamic_slice(current_full_x, (0, 0), (batch_size, current_index))
        # Option 3: Slice fixed context window (if model needs fixed input length)
        # context_length = config.context_length # Get context length
        # start_idx = jnp.maximum(0, current_index - context_length)
        # input_sequence = jax.lax.dynamic_slice(current_full_x, (0, start_idx), (batch_size, current_index - start_idx))
        # Need padding if current_index < context_length

        # Let's assume Option 1 for simplicity here, adjust if needed based on your model
        input_sequence = current_full_x
        out = jitted_stack(input_sequence)  # Shape: [batch, final_len, vocab_size]

        # --- Logit Selection ---
        # Get logits corresponding to the prediction for the *next* token (at current_index)
        # These are typically the output logits at position current_index - 1
        # Use dynamic_slice to get logits at the correct time step: [batch, 1, vocab_size]
        logits_slice = jax.lax.dynamic_slice(
            out, (0, current_index - 1, 0), (batch_size, 1, vocab_size)
        )
        # Squeeze to get [batch, vocab_size]
        last_token_logits = jnp.squeeze(logits_slice, axis=1)

        # Find the best next token: [batch]
        next_token = jnp.argmax(last_token_logits, axis=-1)
        next_token = next_token.astype(jnp.int32)

        # --- State Update ---
        # Update the sequence array at the current index using dynamic_update_slice
        updated_full_x = jax.lax.dynamic_update_slice(
            current_full_x, next_token[:, None], (0, current_index)
        )

        # Return new carry and the collected token
        return (updated_full_x, current_index + 1), next_token

    # Define the function that runs the scan loop and jit it
    @jax.jit
    def run_generation_scan(initial_carry_tuple):
        final_carry, collected_next_tokens = jax.lax.scan(
            generation_step, initial_carry_tuple, None, length=MAX_NEW_TOKEN
        )
        final_x = final_carry[0]  # The full sequence array
        # Optionally return collected_next_tokens as well if needed
        return final_x

    # Compile the scan function before timing
    print("Compiling JAX scan...")
    final_x_compiled = run_generation_scan(initial_carry_val)
    final_x_compiled.block_until_ready()  # Wait for compilation
    print("Compilation complete.")

    # --- Time the jitted scan execution ---
    jax_start = perf_counter()
    final_x = run_generation_scan(initial_carry_val)
    final_x.block_until_ready()
    jax_end = perf_counter()
    # The result 'final_x' now has shape [batch, final_len]
    x = final_x  # Update x to the full generated sequence
    print(f"JAX time (scan): {jax_end - jax_start:.4f} seconds")

    # --- Torch loop ---
    # (Ensure torch loop generates the same number of tokens for comparison)
    torch_start = perf_counter()
    with torch.no_grad():
        current_torch_x = torch_x  # Start with initial torch input
        for _ in tqdm(range(MAX_NEW_TOKEN), desc="Torch"):
            output_logits = torch_xlstm_stack(current_torch_x)
            last_token_logits = output_logits[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1)
            next_token = next_token.to(torch.int32)
            current_torch_x = torch.cat([current_torch_x, next_token[:, None]], dim=1)
        torch_x = current_torch_x  # Update torch_x to the full generated sequence
    torch_end = perf_counter()
    print(f"Torch time: {torch_end - torch_start:.4f} seconds")

    execution_difference = torch_end - torch_start - (jax_end - jax_start)
    diff_percentage = ((torch_end - torch_start) / (jax_end - jax_start) - 1) * 100
    print(
        f"Execution time difference: {execution_difference:.4f} seconds ({diff_percentage:.2f}%)"
    )

    # --- Comparison ---
    # Now both torch_x and x (from JAX) should have shape [batch, initial_len + MAX_NEW_TOKEN]
    print(f"JAX final shape: {x.shape}")
    print(f"Torch final shape: {torch_x.shape}")

    equal = np.allclose(
        torch_x.clone().detach().cpu().numpy(),
        np.array(jax.device_get(x)),
        rtol=1e-4,
        atol=1e-4,
    )
    print(f"The outputs are equal: {equal}")

    # check the equality per sequence
    for i in range(torch_x.shape[0]):
        equal = np.allclose(
            torch_x[i].clone().detach().cpu().numpy(),
            np.array(jax.device_get(x[i])),
            rtol=1e-4,
            atol=1e-4,
        )
        print(f"The outputs are equal for sequence {i}: {equal}")
        print(
            f"Predicted torch sequence {i}: {torch_x[i].clone().detach().cpu().numpy()}"
        )
        print(f"Predicted jax sequence {i}: {np.array(jax.device_get(x[i]))}")


if __name__ == "__main__":
    test_nnx_model_matches_torch_one()
