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
@misc{thiombiano2025xlstmjax,
title={xLSTM-JAX Port},
author={Thiombiano, Abdoul Majid O.},
year={2024},
url={https://github.com/thiomajid/xlstm_jax/},
}
```
