import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from tests.utils import create_jax_model
from xlstm_jax.inference import GenerationCarry, generate_sequence_scan

if __name__ == "__main__":
    model = create_jax_model()
    config = model.config

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    text = "The quick brown fox jumps over the lazy dog"
    input_ids: jnp.ndarray = tokenizer(text, return_tensors="jax")["input_ids"]

    # run inference
    outputs = model(input_ids)
    print(f"First pass output shape: {outputs.shape}")

    # Autoregressive generation
    temperature = 0.7
    max_new_tokens = 20
    greedy = True

    batch_size = input_ids.shape[0]
    initial_len = input_ids.shape[1]
    total_length = initial_len + max_new_tokens
    full_x_init = jnp.zeros((batch_size, total_length), dtype=jnp.int32)
    full_x_init = full_x_init.at[:, :initial_len].set(input_ids)
    key = jax.random.key(123)
    initial_carry: GenerationCarry = (
        full_x_init,
        initial_len,
        key,
    )

    sequence = generate_sequence_scan(
        model,
        initial_carry,
        max_new_tokens=max_new_tokens,
        vocab_size=config.vocab_size,
        temperature=temperature,
        greedy=greedy,
    )

    print(f"Generated sequence shape: {sequence.shape}")
    print(f"Generated sequence: {sequence}")
    print(f"Generated sequence: {tokenizer.decode(sequence[0, initial_len:])}")
