# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Differential check of RWKV-7 against FLA and a float64 oracle, on identical inputs.

    python benchmarks/rwkv7_check.py op --torch-python VENV/bin/python --out results/x.json
    python benchmarks/rwkv7_check.py layer --torch-python VENV/bin/python --out results/y.json

``op``: for each case (odd and long ``T``, head dims 64/128, ``K != V``, packed documents as our
``segment_ids`` against FLA's ``cu_seqlens``, initial and final state, bf16 and fp32, RWKV-7 gates
and strong decays) every implementation of ``kind="rwkv7"`` (``reference``, ``chunked``,
``mosaic_gpu`` default / bounded / ``HIGHEST``) and FLA's ``chunk_rwkv7`` (``torch_rwkv7_check.py``:
exact defaults and the configuration FLA's ``RWKV7Attention`` uses) get the same inputs and the same
output and final-state cotangents. Reported per tensor (``o``, final state, ``dr, dw, dk, dv, da,
db, dh0``): max |error| / max |reference| and max |error|, against the float64 oracle and
between ours and FLA.

``layer``: our ``RWKV7TimeMix`` and ``RWKV7ChannelMix`` against FLA's ``RWKV7Attention``
(``fuse_norm`` False and True) and ``RWKV7FeedForward`` with identical random weights (forward).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla import ops  # noqa: E402

_TORCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_rwkv7_check.py")
_LOWER = -0.6065306597126334
_GRADS = ("r", "k", "v", "w", "a", "b", "h0")  # our argument order (q = r)

OP_CASES = [
    dict(name="dense_odd_T999", batch=2, seq_len=999, heads=4, key_dim=64, value_dim=64),
    dict(name="dense_K128_T333", batch=1, seq_len=333, heads=2, key_dim=128, value_dim=128),
    dict(name="K64_V128_T257", batch=1, seq_len=257, heads=2, key_dim=64, value_dim=128),
    dict(name="packed_4docs_T1024", batch=1, seq_len=1024, heads=4, key_dim=64, value_dim=64, docs=(100, 333, 1, 590)),
    dict(name="long_T4096", batch=1, seq_len=4096, heads=2, key_dim=64, value_dim=64),
    dict(name="fp32_T257", batch=1, seq_len=257, heads=2, key_dim=64, value_dim=64, dtype="float32"),
    dict(name="strong_decay_T300", batch=1, seq_len=300, heads=2, key_dim=64, value_dim=64, gates="strong"),
]

# name -> (impl, options); the bounded mode only where the gates respect its bound.
_OURS = {
    "reference": ("reference", {}),
    "chunked": ("chunked", {"chunk_size": 16}),
    "mosaic_gpu": ("mosaic_gpu", {}),
    "mosaic_gpu_bounded": ("mosaic_gpu", {"lower_bound": -0.61}),
    "mosaic_gpu_highest": ("mosaic_gpu", {"precision": lax.Precision.HIGHEST}),
}


def _round(x: np.ndarray, dtype: str) -> np.ndarray:
    return np.asarray(jnp.asarray(x, jnp.float32).astype(dtype).astype(jnp.float32))


def op_inputs(case: dict) -> dict[str, np.ndarray]:
    """RWKV-7's gates: ``w = -0.6065 sigmoid(.)``, ``a = -kk``, ``b = kk * sigmoid(.)``, ``kk`` unit norm."""
    rng = np.random.default_rng(zlib.crc32(case["name"].encode()))
    b_, t_, h_, kd, vd = (case[n] for n in ("batch", "seq_len", "heads", "key_dim", "value_dim"))
    dtype = case.get("dtype", "bfloat16")
    shape = (b_, t_, h_, kd)
    n = len(case.get("docs") or ()) or b_
    if case.get("gates", "rwkv7") == "strong":  # some channels decay by e^-20 per token
        w = -np.exp(1.5 * rng.standard_normal(shape))
    else:
        w = _LOWER / (1.0 + np.exp(-2.0 * rng.standard_normal(shape)))
    kk = rng.standard_normal(shape)
    kk /= np.linalg.norm(kk, axis=-1, keepdims=True)
    x = dict(
        r=rng.standard_normal(shape), k=0.5 * rng.standard_normal(shape), v=rng.standard_normal((b_, t_, h_, vd)),
        w=w, a=-kk, b=kk / (1.0 + np.exp(-rng.standard_normal(shape))),
        do=rng.standard_normal((b_, t_, h_, vd)),
    )
    x = {key: _round(val, dtype) for key, val in x.items()}
    x["h0"] = (0.3 * rng.standard_normal((n, h_, kd, vd))).astype(np.float32)
    x["dht"] = rng.standard_normal((n, h_, kd, vd)).astype(np.float32)
    if case.get("docs"):
        x["h0"][1:] = 0.0  # ours: the initial state applies to the row's first document only
        x["dht"][:-1] = 0.0  # ours: the final state is the one after the row's last token
        x["cu_seqlens"] = np.concatenate([[0], np.cumsum(case["docs"])]).astype(np.int64)
    return x


def _segment_ids(case: dict) -> np.ndarray | None:
    if not case.get("docs"):
        return None
    return np.concatenate([np.full(d, i, np.int32) for i, d in enumerate(case["docs"])])[None]


def run_ours(case: dict, x: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray] | str:
    impl, options = _OURS[name]
    dtype = case.get("dtype", "bfloat16")
    if impl == "mosaic_gpu" and dtype != "bfloat16":
        return "unsupported: the kernels take bf16 r, k, v"
    if "lower_bound" in options and case.get("gates", "rwkv7") != "rwkv7":
        return "skipped: the gates are below the bound"
    seg = _segment_ids(case)
    seg = None if seg is None else jnp.asarray(seg)
    h0 = jnp.asarray(x["h0"][:1] if case.get("docs") else x["h0"])
    dht = jnp.asarray(x["dht"][-1:] if case.get("docs") else x["dht"])
    args = [jnp.asarray(x[n]).astype(dtype) for n in ("r", "k", "v", "w", "a", "b")] + [h0]
    do = jnp.asarray(x["do"])

    def run(q, k, v, w, a, b, s0):
        return ops.linear_attention("rwkv7", q, k, v, w=w, a=a, b=b, scale=1.0, impl=impl, initial_state=s0,
                                    output_final_state=True, segment_ids=seg, **options)

    def loss(*a):
        o, final = run(*a)
        return jnp.sum(o.astype(jnp.float32) * do) + jnp.sum(final * dht)

    o, final = jax.jit(run)(*args)
    grads = jax.jit(jax.grad(loss, tuple(range(7))))(*args)
    out = dict(o=np.asarray(o.astype(jnp.float32)), final=np.asarray(final))
    out |= {f"d{n}": np.asarray(g.astype(jnp.float32)) for n, g in zip(_GRADS, grads)}
    return out


def _align(case: dict, theirs: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """FLA / oracle results in our layout (packed: the last document's final state, the first's dh0)."""
    out = {}
    for key, val in theirs.items():
        if not key.startswith(prefix + "/"):
            continue
        name = key.split("/", 1)[1]
        if case.get("docs") and name == "final":
            val = val[-1:]
        if case.get("docs") and name == "dh0":
            val = val[:1]
        out[name] = val
    return out


def _errors(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """``tensor -> [max |error| / max |want|, max |error|]``."""
    out = {}
    for name in ("o", "final", "dr", "dw", "dk", "dv", "da", "db", "dh0"):
        if name in got and name in want:
            g, w = np.asarray(got[name], np.float64), np.asarray(want[name], np.float64)
            err = float(np.max(np.abs(g - w)))
            out[name] = [err / max(float(np.max(np.abs(w))), 1e-30), err]
    return out


def _torch(torch_python: str, what: str, inputs: str, out: str, configs: str | None = None) -> dict:
    cmd = [torch_python, _TORCH, what, "--inputs", inputs, "--out", out]
    cmd += ["--configs", configs] if configs else []
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def check_ops(torch_python: str, cases: list[dict], impls: list[str], tmp: str) -> list[dict]:
    rows = []
    for case in cases:
        x = op_inputs(case)
        path = os.path.join(tmp, f"{case['name']}.npz")
        np.savez(path, dtype=np.array(case.get("dtype", "bfloat16")), **x)
        configs = "oracle64,fla_exact" + (",fla_layer" if case.get("gates", "rwkv7") == "rwkv7" else "")
        meta = _torch(torch_python, "op", path, path + ".torch.npz", configs)
        with np.load(path + ".torch.npz") as f:
            theirs = {key: f[key] for key in f.files}
        oracle = _align(case, theirs, "oracle64")
        fla = {c: _align(case, theirs, c) for c in ("fla_exact", "fla_layer") if meta["configs"].get(c) == "ok"}
        for c, res in fla.items():
            rows.append(dict(case=case["name"], impl=c, vs="oracle64", errors=_errors(res, oracle)))
            print(json.dumps(rows[-1]), flush=True)
        for c, status in meta["configs"].items():
            if status != "ok":
                rows.append(dict(case=case["name"], impl=c, status=status))
                print(json.dumps(rows[-1]), flush=True)
        for name in impls:
            res = run_ours(case, x, name)
            if isinstance(res, str):
                rows.append(dict(case=case["name"], impl=name, status=res))
                print(json.dumps(rows[-1]), flush=True)
                continue
            rows.append(dict(case=case["name"], impl=name, vs="oracle64", errors=_errors(res, oracle)))
            print(json.dumps(rows[-1]), flush=True)
            for c, ref in fla.items():
                rows.append(dict(case=case["name"], impl=name, vs=c, errors=_errors(res, ref)))
                print(json.dumps(rows[-1]), flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# layer check
# ---------------------------------------------------------------------------------------------

LAYER_CASES = [
    dict(name="layer0_T100_fp32", layer_idx=0, seq_len=100, batch=2, dtype="float32"),
    dict(name="layer1_T100_fp32", layer_idx=1, seq_len=100, batch=2, dtype="float32"),
    dict(name="layer2_packed_T160_fp32", layer_idx=2, seq_len=160, batch=1, docs=(40, 1, 70, 49), dtype="float32"),
    dict(name="layer1_T37_fp32_recurrent", layer_idx=1, seq_len=37, batch=1, dtype="float32"),
    dict(name="layer1_T128_bf16", layer_idx=1, seq_len=128, batch=1, dtype="bfloat16"),
]
_D, _HEAD, _LAYERS, _LORA, _FF = 256, 64, 4, (32, 32, 32, 64), 896


def _randomize(module, key):
    """Every float leaf redrawn (weights at fan-in scale, other vectors around their init)."""
    leaves, treedef = jax.tree_util.tree_flatten(module)
    keys = jax.random.split(key, len(leaves))
    out = []
    for leaf, k in zip(leaves, keys):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            if leaf.ndim == 2 and leaf.shape[0] > 8:  # a [in, out] weight
                leaf = jax.random.normal(k, leaf.shape) * leaf.shape[0] ** -0.5
            else:
                leaf = leaf + 0.2 * jax.random.normal(k, leaf.shape)
        out.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


def _fla_weights(att, ffn) -> dict[str, np.ndarray]:
    w = {f"att.x_{n}": np.asarray(att.mu[i]) for i, n in enumerate("rwkvag")}
    w |= {"att.k_k": np.asarray(att.k_k), "att.k_a": np.asarray(att.k_a), "att.r_k": np.asarray(att.r_k)}
    for n in ("r", "k", "v", "o"):
        w[f"att.{n}_proj.weight"] = np.asarray(getattr(att, f"{n}_proj").weight).T
    for n in ("w", "a", "v", "g"):
        lora = getattr(att, f"{n}_lora")
        if lora is None:
            continue
        w[f"att.{n}_lora.lora.0.weight"] = np.asarray(lora.down.weight).T
        w[f"att.{n}_lora.lora.2.weight"] = np.asarray(lora.up.weight).T
        if lora.up.bias is not None:
            w[f"att.{n}_lora.lora.2.bias"] = np.asarray(lora.up.bias)
    w |= {"att.norm.weight": np.asarray(att.norm_weight), "att.norm.bias": np.asarray(att.norm_bias)}
    w |= {"ffn.x_k": np.asarray(ffn.x_k), "ffn.key.weight": np.asarray(ffn.key.weight).T,
          "ffn.value.weight": np.asarray(ffn.value.weight).T}
    return w


def check_layers(torch_python: str, cases: list[dict], tmp: str) -> list[dict]:
    from jla.layers.rwkv7 import RWKV7ChannelMix, RWKV7TimeMix

    rows = []
    for case in cases:
        li, dtype = case["layer_idx"], case["dtype"]
        k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(li + 10 * case["seq_len"]), 5)
        common = dict(param_dtype=jnp.float32, compute_dtype=dtype)
        att = RWKV7TimeMix(_D, head_dim=_HEAD, layer_idx=li, n_layers=_LAYERS, key=k1, lora_dims=_LORA, **common)
        ffn = RWKV7ChannelMix(_D, _FF, layer_idx=li, n_layers=_LAYERS, key=k2, **common)
        att, ffn = _randomize(att, k3), _randomize(ffn, k4)
        x = jax.random.normal(k5, (case["batch"], case["seq_len"], _D)).astype(dtype)
        v_first = None if li == 0 else jax.random.normal(jax.random.fold_in(k5, 1), x.shape).astype(dtype)
        seg = _segment_ids(case)
        seg = None if seg is None else jnp.asarray(seg)
        # fp32: f32 matmuls (XLA's default on H100 is TF32, which alone gives ~5e-4 relative errors)
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            att_out, vf_out, _, _ = eqx.filter_jit(lambda a, x, vf: a(x, vf, seg))(att, x, v_first)
            ffn_out, _ = eqx.filter_jit(lambda f, x: f(x, seg))(ffn, x)
        cfg = dict(d_model=_D, head_dim=_HEAD, layer_idx=li, n_layers=_LAYERS, lora=_LORA, norm_eps=1e-5, d_ff=_FF,
                   dtype=dtype)
        data = dict(config=np.array(json.dumps(cfg)), x=np.asarray(x.astype(jnp.float32)), **_fla_weights(att, ffn))
        if v_first is not None:
            data["v_first"] = np.asarray(v_first.astype(jnp.float32))
        if case.get("docs"):
            data["cu_seqlens"] = np.concatenate([[0], np.cumsum(case["docs"])]).astype(np.int64)
        path = os.path.join(tmp, f"{case['name']}.npz")
        np.savez(path, **data)
        _torch(torch_python, "layer", path, path + ".torch.npz")
        with np.load(path + ".torch.npz") as f:
            theirs = {key: f[key] for key in f.files}
        ours = dict(att=np.asarray(att_out.astype(jnp.float32)), v_first=np.asarray(vf_out.astype(jnp.float32)),
                    ffn=np.asarray(ffn_out.astype(jnp.float32)))
        for tag in ("plain", "fused"):
            pairs = {"att": (ours["att"], theirs[f"att_{tag}"]), "v_first": (ours["v_first"], theirs[f"v_first_{tag}"]),
                     "ffn": (ours["ffn"], theirs["ffn"])}
            errs = {}
            for n, (g, w) in pairs.items():
                err = float(np.max(np.abs(g.astype(np.float64) - w)))
                errs[n] = [err / max(float(np.max(np.abs(w))), 1e-30), err]
            rows.append(dict(case=case["name"], fla_fuse_norm=tag == "fused", errors=errs))
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("op", "layer"))
    parser.add_argument("--torch-python", required=True, help="python of a torch environment with FLA")
    parser.add_argument("--cases", default=None, help="comma-separated case names (default: all)")
    parser.add_argument("--impls", default=",".join(_OURS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    pool = OP_CASES if args.what == "op" else LAYER_CASES
    cases = [c for c in pool if args.cases is None or c["name"] in args.cases.split(",")]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        if args.what == "op":
            rows = check_ops(args.torch_python, cases, args.impls.split(","), tmp)
        else:
            rows = check_layers(args.torch_python, cases, tmp)
    result = dict(kind="rwkv7", check=args.what, environment=harness.environment(),
                  metric="per tensor: [max |error| / max |reference|, max |error|]", rows=rows)
    if args.out:
        harness.write_json(args.out, result)


if __name__ == "__main__":
    main()
