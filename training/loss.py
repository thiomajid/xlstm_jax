import jax
import jax.numpy as jnp
import optax
from einops import rearrange


def causal_lm_loss(
    logits: jax.Array,
    labels: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """
    Computes the cross-entropy loss between predicted logits and reference labels
    for the next-token prediction task.
    """

    # shift inputs so that logits at timestep t precedes label at timestep (t+1)
    # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
    shifted_logits = rearrange(logits[..., :-1, :], "b s v -> (b s) v")

    # shape: [batch, seq] -> [batch * (seq-1)]
    shifted_labels = rearrange(labels[..., 1:], "b s -> (b s)")
    shifted_mask = rearrange(attention_mask[..., 1:], "b s -> (b s)").astype(
        shifted_logits.dtype
    )

    # Compute cross-entropy loss
    unreduced_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shifted_logits,
        labels=shifted_labels,
    )

    masked_loss = unreduced_loss * shifted_mask
    normalizer = shifted_mask.sum()
    ce_loss = masked_loss.sum() / jnp.maximum(normalizer, 1e-8)

    return ce_loss
