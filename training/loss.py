import jax
import jax.numpy as jnp
import optax
from einops import rearrange


def causal_lm_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """
    Computes the cross-entropy loss between predicted logits and reference labels
    for the next-token prediction task.
    """
    # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
    shifted_logits = rearrange(logits[..., :-1, :], "b s v -> (b s) v")

    # shape: [batch, seq] -> [batch * (seq-1)]
    shifted_labels = rearrange(labels[..., 1:], "b s -> (b s)")
    shifted_labels = jnp.astype(shifted_labels, logits.dtype)

    # Compute cross-entropy loss
    ce_loss = optax.safe_softmax_cross_entropy(
        logits=shifted_logits,
        labels=shifted_labels,
    ).mean()

    return ce_loss
