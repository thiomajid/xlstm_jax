from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

_dtype_map = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}


def str2dtype(dtype_str: str) -> jnp.dtype:
    """Convert a string representation of a data type to a numpy dtype."""

    if dtype_str not in _dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return _dtype_map[dtype_str]


def is_jax_prng_key(x):
    """Checks if an object is a JAX PRNGKey."""
    # Use the canonical way to check for JAX PRNGKey dtypes
    return isinstance(x, jax.Array) and jax.dtypes.issubdtype(
        x.dtype, jax.dtypes.prng_key
    )


def filter_prng_keys(pytree):
    """Recursively filters out JAX PRNGKeys from a pytree, replacing them with None."""
    return jax.tree.map(
        lambda x: None if is_jax_prng_key(x) else x,
        pytree,
        is_leaf=is_jax_prng_key,  # Treat keys as leaves to prevent recursion into them
    )


def load_model_from_checkpoint(
    model: nnx.Module,
    checkpoint_path: str | Path,
) -> nnx.Module:
    """Load a model from a checkpoint."""
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    print("Created abstract state")

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path).absolute()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path, abstract_state)
    # nnx.replace_by_pure_dict(abstract_state, restored_state)
    merged_model = nnx.merge(graphdef, restored_state)
    print("Merged state with the model.")
    return merged_model
