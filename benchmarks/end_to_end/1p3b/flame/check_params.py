# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shape check: parameter counts of FLA's models (the flame configs here) against the jla presets of the same name.

    <flame venv>/bin/python benchmarks/end_to_end/1p3b/flame/check_params.py <jax python> <jla src dir>

Builds every FLA model on the meta device (no memory, no GPU) and counts its parameters; the jla counts come from
``jla.models.parameter_count`` run by ``<jax python>`` on the CPU. Prints one line per config with the difference.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ["transformer-1p3b", "transformer-swa-1p3b", "gdn-1p3b", "gdn-hd128-1p3b", "kda-1p3b", "mamba2-1p3b",
         "mamba3-1p3b", "rwkv7-1p3b"]
# Configs whose shapes must equal a jla preset (benchmarks/bench_e2e_flame.py JLA_PRESET): a missing preset or a
# parameter-count difference is a failure, not a skip. rwkv7-1p3b is printed but not required to match: the jla
# preset of that name keeps flame's widths and depth, but flame's RWKV-7 uses LayerNorms with bias (and one more
# after the embedding) where the jla RWKV-7 stack uses RMSNorm pre-norm blocks (src/jla/models/presets.py).
REQUIRED = [n for n in NAMES if n != "rwkv7-1p3b"]
JLA_CODE = (
    "import json, sys\n"
    "from jla.models import get_preset, parameter_count\n"
    "out = {}\n"
    "for n in sys.argv[1:]:\n"
    "    try:\n"
    "        c = parameter_count(get_preset(n)); out[n] = [c.total, c.embedding]\n"
    "    except KeyError as e:\n"
    "        out[n] = repr(e)[:80]\n"
    "print(json.dumps(out))\n"
)


def main() -> int:
    jla_python, jla_src = sys.argv[1], sys.argv[2]
    sys.path.insert(0, str(HERE / "fa3_shim"))
    import fla  # noqa: F401  (registers FLA's model classes with transformers)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    env = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": jla_src, "PATH": "/usr/bin:/bin"}
    out = subprocess.run([jla_python, "-c", JLA_CODE, *NAMES], capture_output=True, text=True, env=env, check=True)
    jla = json.loads(out.stdout.strip().splitlines()[-1])
    mismatches = 0
    for name in NAMES:
        config = AutoConfig.from_pretrained(str(HERE / f"{name}.json"))
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        total = sum(p.numel() for p in model.parameters())
        embed = sum(p.numel() for k, p in model.named_parameters() if "embed" in k or "lm_head" in k)
        ref = jla.get(name)
        if isinstance(ref, list):
            diff = total - ref[0]
            mismatches += name in REQUIRED and diff != 0
            print(f"{name:22s} fla {total:>14,d} (embedding {embed:,d})  jla {ref[0]:>14,d} (embedding {ref[1]:,d})"
                  f"  diff {diff:+,d}")
        else:
            missing = name in REQUIRED
            mismatches += missing
            print(f"{name:22s} fla {total:>14,d} (embedding {embed:,d})  jla: no preset ({ref})"
                  + ("  FAIL: a jla preset is required" if missing else ""))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
