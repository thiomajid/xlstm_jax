from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from tests.utils import create_jax_model
from xlstm_jax.utils import filter_prng_keys

if __name__ == "__main__":
    ckpt_dir = Path("./ckpts_demo").absolute()
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)
    checkpointer = ocp.StandardCheckpointer()

    print("Checkpointer ready")

    model = create_jax_model()
    state = nnx.to_pure_dict(nnx.state(model))

    # Save the checkpoint
    filtered_state = filter_prng_keys(state)
    checkpointer.save(ckpt_dir / "state", state)
    checkpointer.wait_until_finished()
    print("Checkpoint saved.")

    # Load the checkpoint
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    print("Created abstract state")

    restored_state = checkpointer.restore(ckpt_dir / "state")

    print(restored_state)

    # restored_state = filter_prng_keys(restored_state)

    # PRNG keys can not be converted to numpy arrays hence we need to filter them out
    # otherwise the test will fail
    # jax.tree.map(np.testing.assert_array_equal, filtered_state, restored_state)

    # merging the state
    nnx.replace_by_pure_dict(abstract_state, restored_state)
    model = nnx.merge(graphdef, restored_state)
    print("Merged state with the model.")
