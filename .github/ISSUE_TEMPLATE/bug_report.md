---
name: Bug report
about: Something does not work, or a result is wrong
labels: bug
---

**What happened**

A short description, and what you expected instead.

**How to reproduce**

The smallest config, command or script that shows it, for example:

```bash
jla train --config configs/tiny-synthetic.toml --set ...
```

```python
from jla import ops
o, state = ops.linear_attention("gated_deltanet", q, k, v, g=g, beta=beta, impl="auto")
```

For a wrong result, say which implementations disagree (`impl="reference"`, `"chunked"`,
`"mosaic_gpu"`) and by how much, with the shapes and dtypes.

**Environment**

- jax-linear-attention version (`jla --version` or `python -c "import jla; print(jla.__version__)"`):
- JAX, jaxlib and CUDA plugin versions (`pip list | grep -i -E "jax|nvidia"`):
- Python version:
- GPU and driver (`nvidia-smi`), or CPU only:
- Number of GPUs and the config's `[mesh]` table, if more than one:

**Logs**

The error and traceback, or the relevant lines of the training log.
