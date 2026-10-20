# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Differential test of the GLA, Mamba-1 and mLSTM ops against their latest upstream kernels.

    python benchmarks/diff_upstream.py --ops gla,mamba1,mlstm \\
        --fla-python VENV_FLA/bin/python --mamba-python VENV_MAMBA/bin/python \\
        --mlstm-python VENV_MLSTM/bin/python --out benchmarks/results/diff_upstream_<date>.json

Every case draws one set of random inputs (bf16-exact values for the bf16 cases) and output
cotangents, then runs:

* the float32 oracle: ``jla.ops.reference`` on the float32 values (``HIGHEST`` matmul precision);
* each of our implementations in the case dtype (``reference``, ``chunked``, ``mosaic_gpu`` and, for
  GLA, ``mosaic_gpu`` with ``pipeline="vector_gate"``), skipping the ones whose ``supports`` rejects
  the call;
* the upstream op in a torch subprocess on the same values and dtype (``torch_upstream_diff.py``):
  FLA's ``chunk_gla`` (packed documents as ``cu_seqlens``), the official ``selective_scan_cuda``
  forward and backward of state-spaces/mamba (no initial state or packed documents upstream: those
  cases compare ours against the oracle only), and NX-AI's TFLA ``mlstm_kernels`` (dense rows, no
  initial state). ``--mlstm-python`` is optional: ``mlstm_kernels`` is not permissively licensed
  and only used as an external baseline through its public API.

For every tensor (``o``, the final state and each input gradient) it records the max abs error and
that error over the oracle's max magnitude, for every implementation and for upstream, and the same
between our ``mosaic_gpu`` result and upstream's. For mLSTM cases that run TFLA it also compares
TFLA with the reference differentiated with the stabiliser and the denominator held constant
(:func:`mlstm_stop_gradient_reference`), and that reference with the exact oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla import ops  # noqa: E402
from jla.ops.reference import MLSTMState  # noqa: E402

F32 = jnp.float32
BF16 = jnp.bfloat16

# --- cases ------------------------------------------------------------------------------------------

GLA_CASES = [
    dict(name="gla T=1000 K=V=128 h0", B=2, T=1000, H=4, K=128, V=128, h0=True),
    dict(name="gla T=4097 K=64 V=128 (gla-125m heads)", B=1, T=4097, H=6, K=64, V=128, h0=False),
    dict(name="gla T=333 K=V=64 h0 strong gates", B=2, T=333, H=4, K=64, V=64, h0=True, gate="strong"),
    dict(name="gla packed docs h0", B=1, T=1100, H=2, K=128, V=128, h0=True, docs=[300, 1, 64, 500, 235]),
    dict(name="gla T=1000 K=V=128 h0 fp32", B=2, T=1000, H=4, K=128, V=128, h0=True, dtype="float32"),
    dict(name="gla packed docs fp32", B=1, T=700, H=2, K=64, V=64, h0=True, docs=[129, 300, 271],
         dtype="float32"),
]
MAMBA1_CASES = [
    dict(name="mamba1 T=1000 D=256 N=16", B=2, T=1000, D=256, N=16),
    dict(name="mamba1 T=4129 D=1536 N=16 (mamba-130m layer)", B=1, T=4129, D=1536, N=16),
    dict(name="mamba1 T=333 D=256 N=8 large dt", B=2, T=333, D=256, N=8, dt="large"),
    dict(name="mamba1 T=1000 D=256 N=16 fp32", B=2, T=1000, D=256, N=16, dtype="float32"),
    dict(name="mamba1 packed docs h0 (no upstream)", B=2, T=600, D=256, N=16, h0=True, docs=[100, 1, 250, 249]),
    dict(name="mamba1 T=1000 D=256 N=32", B=2, T=1000, D=256, N=32),
    dict(name="mamba1 T=4096 D=1024 N=16 (one segment per 512 tokens)", B=1, T=4096, D=1024, N=16),
]
MLSTM_CASES = [
    dict(name="mlstm T=1024 K=256 V=512 (xLSTM-7B heads)", B=1, T=1024, H=2, K=256, V=512),
    dict(name="mlstm T=1000 K=128 V=256", B=2, T=1000, H=2, K=128, V=256),
    dict(name="mlstm T=777 K=V=64 h0 (no upstream)", B=2, T=777, H=2, K=64, V=64, h0=True),
    dict(name="mlstm packed docs (no upstream)", B=1, T=900, H=2, K=128, V=128, docs=[200, 1, 400, 299]),
    dict(name="mlstm T=256 K=V=64", B=1, T=256, H=2, K=64, V=64),
]


def _round(x: np.ndarray, dtype: str) -> np.ndarray:
    """``x`` as float32 values that the case dtype represents exactly."""
    x = np.asarray(x, np.float32)
    return np.asarray(jnp.asarray(x).astype(BF16).astype(F32)) if dtype == "bfloat16" else x


def _segments(docs: list[int], batch: int) -> np.ndarray:
    ids = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(docs)])
    return np.repeat(ids[None], batch, axis=0)


def gla_inputs(case: dict, rng: np.random.Generator) -> dict:
    b, t, h, kd, vd, dtype = case["B"], case["T"], case["H"], case["K"], case["V"], case.get("dtype", "bfloat16")
    x = dict(q=_round(rng.standard_normal((b, t, h, kd)), dtype),
             k=_round(rng.standard_normal((b, t, h, kd)), dtype),
             v=_round(rng.standard_normal((b, t, h, vd)), dtype))
    raw = rng.standard_normal((b, t, h, kd)).astype(np.float32)
    g = np.log(1.0 / (1.0 + np.exp(-raw))) / 16.0  # FLA's test gates: logsigmoid / 16
    if case.get("gate") == "strong":  # 10 % of the channels decay by e^-30 to e^-60 per token
        strong = rng.random((b, t, h, kd)) < 0.1
        g = np.where(strong, -30.0 - 30.0 * rng.random((b, t, h, kd)), g)
    x["g"] = g.astype(np.float32)
    if case.get("h0"):
        x["h0"] = rng.standard_normal((b, h, kd, vd)).astype(np.float32)
    x["do"] = rng.standard_normal((b, t, h, vd)).astype(np.float32)
    x["dht"] = rng.standard_normal((b, h, kd, vd)).astype(np.float32)
    return x


def mamba1_inputs(case: dict, rng: np.random.Generator) -> dict:
    b, t, d, n, dtype = case["B"], case["T"], case["D"], case["N"], case.get("dtype", "bfloat16")
    scale = 2.0 if case.get("dt") == "large" else 0.5
    x = dict(q=_round(rng.standard_normal((b, t, n)), dtype), k=_round(rng.standard_normal((b, t, n)), dtype),
             v=_round(rng.standard_normal((b, t, d)), dtype),
             dt=_round(scale * rng.standard_normal((b, t, d)), dtype),
             z=_round(rng.standard_normal((b, t, d)), dtype))
    x["a"] = -np.exp(rng.uniform(-1.0, 1.5, (d, n))).astype(np.float32)
    x["d"] = rng.standard_normal(d).astype(np.float32)
    lo, hi = (-1.0, 1.0) if case.get("dt") == "large" else (-4.0, -2.0)
    x["dt_bias"] = rng.uniform(lo, hi, d).astype(np.float32)
    if case.get("h0"):
        x["h0"] = rng.standard_normal((b, d, n)).astype(np.float32)
    x["do"] = rng.standard_normal((b, t, d)).astype(np.float32)
    # The official op does not differentiate its last state: a final-state cotangent only where it is not run.
    upstream = not (case.get("h0") or "docs" in case)
    x["dht"] = (np.zeros if upstream else rng.standard_normal)((b, d, n)).astype(np.float32)
    return x


def mlstm_inputs(case: dict, rng: np.random.Generator) -> dict:
    b, t, h, kd, vd, dtype = case["B"], case["T"], case["H"], case["K"], case["V"], case.get("dtype", "bfloat16")
    x = dict(q=_round(rng.standard_normal((b, t, h, kd)), dtype),
             k=_round(rng.standard_normal((b, t, h, kd)), dtype),
             v=_round(rng.standard_normal((b, t, h, vd)), dtype),
             # gate pre-activations (bf16 values: TFLA takes them in the input dtype)
             i=_round(3.0 * rng.standard_normal((b, t, h)), "bfloat16"),
             f=_round(3.0 * rng.standard_normal((b, t, h)) + 2.0, "bfloat16"))
    if case.get("h0"):
        x["c0"] = 0.1 * rng.standard_normal((b, h, kd, vd)).astype(np.float32)
        x["n0"] = 0.1 * rng.standard_normal((b, h, kd)).astype(np.float32)
        x["m0"] = rng.standard_normal((b, h)).astype(np.float32)
    x["do"] = rng.standard_normal((b, t, h, vd)).astype(np.float32)
    # TFLA runs without its last states here: a final-state cotangent only where it is not run.
    upstream = not (case.get("h0") or "docs" in case)
    x["dc"] = (np.zeros if upstream else rng.standard_normal)((b, h, kd, vd)).astype(np.float32)
    x["dn"] = (np.zeros if upstream else rng.standard_normal)((b, h, kd)).astype(np.float32)
    return x


# --- our side ---------------------------------------------------------------------------------------


def _diff_names(op: str, x: dict) -> list[str]:
    names = {"gla": ["q", "k", "v", "g", "h0"], "mamba1": ["q", "k", "v", "dt", "a", "d", "z", "dt_bias", "h0"],
             "mlstm": ["q", "k", "v", "i", "f", "c0", "n0", "m0"]}[op]
    return [n for n in names if n in x]


def _call(op: str, impl: str, vals: dict, seg, extra: dict):
    """``(o, final)`` of our op; ``final`` is a tuple of arrays."""
    if op == "gla":
        o, ht = ops.linear_attention("gla", vals["q"], vals["k"], vals["v"], g=vals["g"],
                                     initial_state=vals.get("h0"), output_final_state=True, segment_ids=seg,
                                     impl=impl, **extra)
        return o, (ht,)
    if op == "mamba1":
        o, ht = ops.linear_attention("mamba1", vals["q"], vals["k"], vals["v"], dt=vals["dt"], a=vals["a"],
                                     d=vals["d"], z=vals["z"], dt_bias=vals["dt_bias"], dt_softplus=True,
                                     initial_state=vals.get("h0"), output_final_state=True, segment_ids=seg,
                                     impl=impl, **extra)
        return o, (ht,)
    h0 = MLSTMState(vals["c0"], vals["n0"], vals["m0"]) if "c0" in vals else None
    o, st = ops.linear_attention("mlstm", vals["q"], vals["k"], vals["v"], g=jax.nn.log_sigmoid(vals["f"]),
                                 i=vals["i"], initial_state=h0, output_final_state=True, segment_ids=seg,
                                 impl=impl, **extra)
    return o, (st.c, st.n)


def run_ours(op: str, impl: str, x: dict, dtype, seg, extra: dict) -> dict | str:
    """Outputs and gradients of one implementation (float32 numpy), or why it does not run the call."""
    names = _diff_names(op, x)
    act = {"gla": ("q", "k", "v"), "mamba1": ("q", "k", "v", "dt", "z"), "mlstm": ("q", "k", "v")}[op]
    vals = {n: jnp.asarray(x[n]).astype(dtype if n in act else F32) for n in names}
    seg = None if seg is None else jnp.asarray(seg)
    cts = {"gla": ("do", "dht"), "mamba1": ("do", "dht"), "mlstm": ("do", "dc", "dn")}[op]
    ct = [jnp.asarray(x[n]) for n in cts]

    def loss(*args):
        o, final = _call(op, impl, dict(zip(names, args, strict=True)), seg, extra)
        total = jnp.sum(o.astype(F32) * ct[0])
        for f, c in zip(final, ct[1:], strict=True):
            total = total + jnp.sum(f.astype(F32) * c)
        return total, (o, final)

    try:
        grads, (o, final) = jax.jit(jax.grad(loss, argnums=tuple(range(len(names))), has_aux=True))(
            *(vals[n] for n in names))
    except (ValueError, NotImplementedError) as err:
        return f"{type(err).__name__}: {str(err)[:200]}"
    out = {"o": np.asarray(o, np.float32)}
    for label, f in zip(("final", "final_n"), final):
        out[label] = np.asarray(f, np.float32)
    for n, g in zip(names, grads, strict=True):
        out["d" + n] = np.asarray(g, np.float32)
    return out


def mlstm_stop_gradient_reference(x: dict) -> dict:
    """The float32 mLSTM of ``jla.ops.reference.mlstm`` (dense rows, no initial state, ``do`` the only
    cotangent) with the stabiliser ``m`` and the denominator ``max(|q n|, exp(-m)) + eps`` held constant
    under differentiation (``stop_gradient``).

    The forward is the exact op's; only the gradient differs. TFLA's backward (``mlstm_kernels``) is
    compared against this as well as against the exact oracle (``upstream_vs_stop_gradient_reference``).
    """
    sg = jax.lax.stop_gradient
    names = ("q", "k", "v", "i", "f")
    do = jnp.asarray(x["do"], F32)

    def mlstm(q, k, v, i, f):
        batch, _, heads, key_dim = q.shape
        state0 = (jnp.zeros((batch, heads, key_dim, v.shape[-1]), F32), jnp.zeros((batch, heads, key_dim), F32),
                  jnp.full((batch, heads), -1e30, F32))

        def step(carry, xs):
            c, n, m = carry
            q_t, k_t, v_t, g_t, i_t = xs
            m_new = jnp.maximum(g_t + m, i_t)
            forget, write = jnp.exp(g_t + sg(m) - sg(m_new)), jnp.exp(i_t - sg(m_new))
            c = forget[..., None, None] * c + write[..., None, None] * k_t[..., :, None] * v_t[..., None, :]
            n = forget[..., None] * n + write[..., None] * k_t
            num = jnp.einsum("bhk,bhkv->bhv", q_t, c)
            den = jnp.maximum(jnp.abs(jnp.sum(q_t * n, axis=-1)), jnp.exp(jnp.minimum(-m_new, 80.0))) + 1e-6
            return (c, n, m_new), num / sg(den)[..., None]

        xs = tuple(jnp.moveaxis(t, 1, 0) for t in (q * key_dim**-0.5, k, v, jax.nn.log_sigmoid(f), i))
        return jnp.moveaxis(jax.lax.scan(step, state0, xs)[1], 0, 1)

    def loss(*args):
        o = mlstm(*args)
        return jnp.sum(o * do), o

    with jax.default_matmul_precision("highest"):
        grads, o = jax.jit(jax.grad(loss, argnums=tuple(range(len(names))), has_aux=True))(
            *(jnp.asarray(x[n], F32) for n in names))
    return {"o": np.asarray(o, np.float32), **{"d" + n: np.asarray(g, np.float32) for n, g in zip(names, grads)}}


# --- upstream side ----------------------------------------------------------------------------------


def run_upstream(python: str, op: str, x: dict, case: dict, tmp_dir: str | None = None) -> dict | str:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_upstream_diff.py")
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        inp, outp = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(inp, **x)
        meta = dict(case, op=op)
        proc = subprocess.run([python, script, inp, outp, json.dumps(meta)], capture_output=True, text=True,
                              timeout=1800, env=dict(os.environ))
        if proc.returncode != 0:
            return harness.error_summary(proc.stderr or proc.stdout, 600)
        info = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        res = {n: np.asarray(a, np.float32) for n, a in np.load(outp).items()}
        res["_info"] = json.loads(info[-1]) if info else {}
        return res


def _err(got: np.ndarray, want: np.ndarray) -> dict:
    if got.shape != want.shape:
        return dict(error=f"shape {got.shape} vs {want.shape}")
    diff = np.abs(got - want)
    scale = float(np.abs(want).max())
    finite = bool(np.isfinite(got).all())
    return dict(max_abs=float(diff.max()), rel=float(diff.max()) / max(scale, 1e-30), finite=finite)


def compare(res: dict, ref: dict) -> dict:
    return {n: _err(res[n], ref[n]) for n in ref if not n.startswith("_") and n in res}


def run_case(op: str, case: dict, seed: int, uses: dict, tmp_dir: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    x = {"gla": gla_inputs, "mamba1": mamba1_inputs, "mlstm": mlstm_inputs}[op](case, rng)
    seg = None if "docs" not in case else _segments(case["docs"], case["B"])
    dtype = F32 if case.get("dtype") == "float32" else BF16
    with jax.default_matmul_precision("highest"):
        oracle = run_ours(op, "reference", x, F32, seg, {})
    assert isinstance(oracle, dict), oracle
    impls = [("reference", {}), ("chunked", {}), ("mosaic_gpu", {})]
    if op == "gla":
        impls.append(("mosaic_gpu", {"pipeline": "vector_gate"}))
    row = dict(op=op, case=case, seed=seed, impls={})
    ours = {}
    for impl, extra in impls:
        label = impl + "".join(f":{v}" for v in extra.values())
        res = run_ours(op, impl, x, dtype, seg, extra)
        if isinstance(res, str):
            row["impls"][label] = dict(skipped=res)
            continue
        ours[label] = res
        row["impls"][label] = compare(res, oracle)
        print(op, case["name"], label, json.dumps({n: round(e.get("rel", -1), 6) for n, e in
                                                    row["impls"][label].items()}), flush=True)
    python = uses.get(op)
    if python and "docs" in case and op != "gla":
        row["upstream"] = dict(skipped="upstream has no packed documents")
    elif python and case.get("h0") and op != "gla":
        row["upstream"] = dict(skipped="upstream has no initial state")
    elif python:
        up = run_upstream(python, op, x, case, tmp_dir)
        if isinstance(up, str):
            row["upstream"] = dict(error=up)
            print(op, case["name"], "upstream error", up[:300], flush=True)
        else:
            row["upstream_info"] = up.pop("_info", {})
            row["upstream"] = compare(up, oracle)
            if "mosaic_gpu" in ours:
                row["mosaic_gpu_vs_upstream"] = compare(ours["mosaic_gpu"], up)
            print(op, case["name"], "upstream", json.dumps({n: round(e.get("rel", -1), 6) for n, e in
                                                             row["upstream"].items()}), flush=True)
            if op == "mlstm":  # TFLA's backward against the gradient with m and the denominator held constant
                sg = mlstm_stop_gradient_reference(x)
                row["stop_gradient_reference_vs_oracle"] = compare(sg, oracle)
                row["upstream_vs_stop_gradient_reference"] = compare(up, sg)
                for label in ("stop_gradient_reference_vs_oracle", "upstream_vs_stop_gradient_reference"):
                    print(op, case["name"], label, json.dumps({n: round(e.get("rel", -1), 6) for n, e in
                                                               row[label].items()}), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="gla,mamba1,mlstm")
    parser.add_argument("--fla-python", default=None)
    parser.add_argument("--mamba-python", default=None)
    parser.add_argument("--mlstm-python", default=None)
    parser.add_argument("--cases", default=None, help="substring filter on case names")
    parser.add_argument("--tmp-dir", default=None, help="directory for the inputs handed to the torch side")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    uses = {"gla": args.fla_python, "mamba1": args.mamba_python, "mlstm": args.mlstm_python}
    table = {"gla": GLA_CASES, "mamba1": MAMBA1_CASES, "mlstm": MLSTM_CASES}
    rows = []
    for op in args.ops.split(","):
        for n, case in enumerate(table[op]):
            if args.cases and args.cases not in case["name"]:
                continue
            rows.append(run_case(op, case, 1000 + n, uses, args.tmp_dir))
            if args.out:
                harness.write_json(args.out, dict(environment=harness.environment(), rows=rows))


if __name__ == "__main__":
    main()
