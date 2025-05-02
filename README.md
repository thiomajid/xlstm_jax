# xLSTM-JAX

This repository contains a JAX/Flax port of the original xLSTM implementation, initially developed in PyTorch by NXAI GmbH. It includes implementations for both the scalar LSTM (sLSTM) and matrix LSTM (mLSTM) units proposed in the xLSTM paper.

## References

The core concepts are based on the following paper:

```bibtex
@article{xlstm,
title={xLSTM: Extended Long Short-Term Memory},
author={Beck, Maximilian and P{"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
journal={arXiv preprint arXiv:2405.04517},
year={2024}
}
```

The original PyTorch implementation can be found here:

```bibtex
@misc{xlstm-pytorch,
title={xlstm-pytorch},
author={NXAI GmbH},
year={2024},
url={https://github.com/NX-AI/xlstm/},
}
```

## Citation

If you use this JAX/Flax implementation in your research, please cite it as follows:

```bibtex
@misc{xlstm-jax-port,
title={xLSTM-JAX Port},
author={Thiombiano, Abdoul Majid O.},
year={2024},
url={https://github.com/thiomajid/xlstm_jax/},
}
```

# o4-mini proposal

def filter_prng_keys(pytree):
"""Recursively drop entire 'rngs' sub‑trees and null out any PRNGKey leaves.""" # If this node is a dict, we rebuild it without any 'rngs' keys
if isinstance(pytree, dict):
return {
k: filter_prng_keys(v)
for k, v in pytree.items()
if k != "rngs"
} # Otherwise map over the leaves and null out any PRNGKey arrays
return jax.tree_map(
lambda x: None
if hasattr(x, "dtype") and str(x.dtype).startswith("key<")
else x,
pytree,
)
