import jax
import jax.numpy as jnp
from dacite import Config as DaciteConfig
from dacite import from_dict
from flax import nnx
from omegaconf import OmegaConf

from training.loss import causal_lm_loss
from training.utils.array import create_mesh
from xlstm_jax import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.inference import generate_sequence_scan

if __name__ == "__main__":
    # create new model
    xlstm_cfg = """ 
    vocab_size: 128
    context_length: 32      
    num_blocks: 4 #!
    embedding_dim: 64 #!
    tie_weights: false
    slstm_at: "all"
    mlstm_block:
        mlstm:
            conv1d_kernel_size: 4
            qkv_proj_blocksize: 4
            num_heads: 4
    
    slstm_block:
        slstm:
            conv1d_kernel_size: 4
            num_heads: 4
        feedforward:
            proj_factor: 1.2
            act_fn: "selu"
    
    """
    cfg = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(
        data_class=xLSTMLMModelConfig,
        data=OmegaConf.to_container(cfg),
        config=DaciteConfig(strict=True),
    )

    mesh = create_mesh((1, 1, 1), ("dp", "tp", "debug"))
    rngs = nnx.Rngs(123)
    dtype = jnp.bfloat16
    param_dtype = jnp.float32

    model = xLSTMLMModel(
        cfg,
        mesh=mesh,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    @nnx.jit
    def run(model, x):
        return model(x)

    x_in = jax.random.randint(
        rngs(),
        shape=(2, 32),
        minval=0,
        maxval=cfg.vocab_size,
    )

    y = model(x_in)
    print(y.shape)

    y_jit = run(model, x_in)

    print(jnp.allclose(y, y_jit))

    print(causal_lm_loss(y, x_in))

    randint = jax.random.randint(
        rngs(),
        shape=(4, 10),
        minval=0,
        maxval=cfg.vocab_size,
        dtype=jnp.int32,
    )

    max_new_tokens = 10
    seq_len = randint.shape[1]

    full_x_init = jnp.zeros(
        shape=(randint.shape[0], seq_len + max_new_tokens),
        dtype=randint.dtype,
    )

    full_x_init = full_x_init.at[:, :seq_len].set(randint)
    carry = (full_x_init, seq_len, rngs())
    sequences = generate_sequence_scan(
        model,
        carry,
        max_new_tokens,
        cfg.vocab_size,
        0.675,
        False,
    )

    print(sequences.shape)
