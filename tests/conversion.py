from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

from xlstm_jax.inference import generate_sequence_scan

from .utils import create_jax_model, create_torch_model


def test_nnx_model_matches_torch_one():
    torch_xlstm_stack, xlstm_stack = create_torch_model(), create_jax_model()

    xlstm_stack.eval()
    torch_xlstm_stack = torch_xlstm_stack.eval()

    config = torch_xlstm_stack.config
    torch_x = torch.randint(1, config.vocab_size, (10, 1), dtype=torch.int32)
    x = jnp.array(torch_x.clone().detach().cpu().numpy(), dtype=jnp.int32)

    print("Initializing the jax model with the torch model parameters")
    xlstm_stack.load_from_torch(torch_xlstm_stack)
    print("Initialized the jax model with the torch model parameters")

    MAX_NEW_TOKEN = 10
    TEMPERATURE = 0.7
    SEED = 42
    USE_GREEDY = False  # Set to True for greedy decoding

    # --- Setup for JAX Scan ---
    initial_len = x.shape[1]
    final_len = initial_len + MAX_NEW_TOKEN
    batch_size = x.shape[0]
    full_x_init = jnp.zeros((batch_size, final_len), dtype=jnp.int32)
    full_x_init = full_x_init.at[:, :initial_len].set(x)
    key = jax.random.key(SEED)
    initial_carry_val = (full_x_init, initial_len, key)
    vocab_size = config.vocab_size

    # --- Compile and Time the jitted scan execution ---
    print("Compiling JAX generation function (includes scan)...")
    # Run once to compile with specific static arguments
    final_x_compiled = generate_sequence_scan(
        xlstm_stack,  # Pass the model object directly
        initial_carry_val,
        max_new_tokens=MAX_NEW_TOKEN,
        vocab_size=vocab_size,
        temperature=TEMPERATURE,
        greedy=USE_GREEDY,
    )
    final_x_compiled.block_until_ready()  # Wait for compilation
    print("Compilation complete.")

    jax_start = perf_counter()
    # Call the jitted function again for timing
    final_x = generate_sequence_scan(
        xlstm_stack,  # Pass the model object directly
        initial_carry_val,
        max_new_tokens=MAX_NEW_TOKEN,
        vocab_size=vocab_size,
        temperature=TEMPERATURE,
        greedy=USE_GREEDY,
    )
    final_x.block_until_ready()
    jax_end = perf_counter()
    x = final_x
    print(f"JAX time (scan, greedy={USE_GREEDY}): {jax_end - jax_start:.4f} seconds")

    # --- Torch loop ---
    torch.manual_seed(SEED)
    torch_start = perf_counter()
    with torch.no_grad():
        current_torch_x = torch_x.clone()
        for _ in tqdm(range(MAX_NEW_TOKEN), desc="Torch"):
            output_logits = torch_xlstm_stack(current_torch_x)
            last_token_logits = output_logits[:, -1, :]

            if USE_GREEDY:
                next_token = torch.argmax(last_token_logits, dim=-1)
                next_token = next_token.unsqueeze(-1)  # Add dim for cat
            else:
                scaled_logits = last_token_logits / TEMPERATURE
                probabilities = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1
                )  # Shape: [batch, 1]

            next_token = next_token.to(torch.int32)
            current_torch_x = torch.cat([current_torch_x, next_token], dim=1)
        torch_x = current_torch_x
    torch_end = perf_counter()
    print(f"Torch time (greedy={USE_GREEDY}): {torch_end - torch_start:.4f} seconds")

    execution_difference = torch_end - torch_start - (jax_end - jax_start)
    diff_percentage = (
        ((torch_end - torch_start) / (jax_end - jax_start) - 1) * 100
        if (jax_end - jax_start) != 0
        else float("inf")
    )
    print(
        f"Execution time difference: {execution_difference:.4f} seconds ({diff_percentage:.2f}%)"
    )

    # --- Comparison ---
    print(f"JAX final shape: {x.shape}")
    print(f"Torch final shape: {torch_x.shape}")

    expected_equal = USE_GREEDY
    equal = np.allclose(
        torch_x.clone().detach().cpu().numpy(),
        np.array(jax.device_get(x)),
        rtol=1e-4,
        atol=1e-4,
    )
    print(f"The outputs are equal (expected {expected_equal}): {equal}")

    # for i in range(torch_x.shape[0]):
    #     seq_equal = np.allclose(
    #         torch_x[i].clone().detach().cpu().numpy(),
    #         np.array(jax.device_get(x[i])),
    #         rtol=1e-4,
    #         atol=1e-4,
    #     )
    #     print(
    #         f"The outputs are equal for sequence {i} (expected {expected_equal}): {seq_equal}"
    #     )
    #     if not seq_equal or i < 2:
    #         print(
    #             f"Predicted torch sequence {i}: {torch_x[i].clone().detach().cpu().numpy()}"
    #         )
    #         print(f"Predicted jax sequence {i}: {np.array(jax.device_get(x[i]))}")


if __name__ == "__main__":
    test_nnx_model_matches_torch_one()
