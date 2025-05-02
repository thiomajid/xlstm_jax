import json

from flax import nnx

from xlstm_jax import xLSTMLMModel
from xlstm_jax.utils import load_model_from_checkpoint, parse_xlstm_config_dict

if __name__ == "__main__":
    with open("./outputs/2025-05-02/20-28-10/artifacts/config.json") as f:
        config_dict = json.load(f)
    config = parse_xlstm_config_dict(config_dict)
    model = xLSTMLMModel(config, rngs=nnx.Rngs(0))
    restored_model = load_model_from_checkpoint(
        model, checkpoint_path="./outputs/2025-05-02/20-28-10/artifacts/state"
    )
