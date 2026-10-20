# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Differential check of KDA / GDN-2: every jla implementation (and FLA's KDA) on identical inputs.

    python benchmarks/diff_vector_gate.py --kind kda --torch-python VENV/bin/python [--cases a,b] [--out f.json]
    python benchmarks/diff_vector_gate.py --kind gdn2 [--cases a,b] [--out f.json]

Each case draws its inputs once (bf16 ``q, k, v`` unless the case says float32, f32 gates) and runs:

* ``oracle64``: the recurrence written directly from the equations in NumPy float64, forward only. KDA:
  ``S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T`` (Eq. 7 of arXiv 2605.22791); GDN-2:
  ``S_t = (I - k_t (b_t * k_t)^T) D_t S_{t-1} + k_t (w_t * v_t)^T`` (Eq. 9-10), ``o_t = scale S_t^T q_t``;
* ``reference``: ``jla.ops.reference`` in float32 (inputs upcast), forward and autodiff gradients; it is the
  oracle of the gradients;
* ``chunked`` and ``mosaic_gpu``: ``jla.ops.linear_attention(..., impl=...)`` (``mosaic_gpu`` needs bf16);
* ``fla`` (KDA): FLA's ``chunk_kda`` (``torch_kda_diff.py``, run with ``--torch-python``) on the same values,
  with ``cu_seqlens`` where jla takes ``segment_ids``, and FLA's in-kernel gate / beta sigmoid / L2 norm in the
  gate cases (jla computes the same gate in XLA); in the ``flashkda_*`` cases also FlashKDA's fused forward
  (``flash_kda.fwd``, outputs only; run these with a torch environment that has FlashKDA).

Gradients are the VJP of ``(o, final_state)`` for fixed random cotangents, with respect to every input
(``q, k, v``, the gates, the raw gate parameters ``A_log, dt_bias`` in the gate cases, ``h0``). For each tensor
the result records ``max`` (max abs error / max abs of the float32 reference) and ``rms`` (relative L2 error) of
each implementation against the reference, of the reference against ``oracle64`` (forward), and of mosaic_gpu
against FLA. ``causality`` cases change the gates after position ``P`` and check that outputs before ``P`` and
the gradients of a loss on them do not change (bit for bit, and by how much otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla import ops  # noqa: E402
from jla.ops import reference as ref  # noqa: E402

F32, BF16 = jnp.float32, jnp.bfloat16
LOWER_BOUND = -5.0

# name -> shape, gates and options. ``cu`` packs documents (one row); ``gate`` picks the gate inputs.
CASES = {
    "dense_odd_T": dict(B=2, T=1000, H=4, K=128, V=128, gate="plain", h0=True),
    "packed_docs": dict(B=1, T=2048, H=4, K=128, V=128, gate="plain", cu=(0, 37, 300, 301, 1024, 1500, 2048)),
    # packed documents with an initial state for the row's first document (FLA: h0 of the first sequence, zeros
    # for the others)
    "packed_docs_h0": dict(B=1, T=2048, H=4, K=128, V=128, gate="plain", cu=(0, 37, 300, 301, 1024, 1500, 2048),
                           h0=True),
    "extreme_gates": dict(B=1, T=777, H=2, K=128, V=128, gate="extreme", h0=True),
    "k64_l2norm": dict(B=2, T=513, H=4, K=64, V=64, gate="plain", h0=True, l2norm=True),
    "lower_bound_gate": dict(B=1, T=1000, H=4, K=128, V=128, gate="lower_bound", h0=True, l2norm=True),
    "softplus_gate": dict(B=1, T=1000, H=4, K=128, V=128, gate="softplus", h0=True, l2norm=True),
    "float32": dict(B=1, T=300, H=2, K=64, V=64, gate="plain", h0=True, dtype="float32"),
    "long_T": dict(B=1, T=8192, H=2, K=128, V=128, gate="plain", h0=True),
    # grouped value heads (FLA's GVA: v, g, beta on HV heads, q, k on H); jla repeats q, k as its layers do
    "grouped_value_heads": dict(B=1, T=777, H=2, HV=4, K=128, V=128, gate="plain", h0=True),
    # FlashKDA's contract (bf16 raw gate and beta logits, -5 lower bound, L2 norm): FlashKDA's forward too
    "flashkda_contract": dict(B=1, T=1000, H=4, K=128, V=128, gate="lower_bound", l2norm=True, flashkda=True),
    "flashkda_packed": dict(B=1, T=2048, H=4, K=128, V=128, gate="lower_bound", l2norm=True, flashkda=True,
                            cu=(0, 37, 300, 301, 1024, 1500, 2048)),
}
CAUSALITY = {
    # future gates (t >= P, P inside a chunk) set to the -5 bound, to a huge decay, or to no decay
    "causality_bound": dict(B=1, T=200, H=2, K=128, V=128, P=150, edit=-4.95),
    "causality_huge": dict(B=1, T=200, H=2, K=128, V=128, P=150, edit=-60.0),
    "causality_zero": dict(B=1, T=200, H=2, K=128, V=128, P=150, edit=0.0),
    # P before the 16th token of a 32-token diagonal block (chunk 2 = tokens 128-191, block 128-159)
    "causality_mid_bound": dict(B=1, T=200, H=2, K=128, V=128, P=135, edit=-4.95),
}
KDA_ONLY = ("grouped_value_heads", "flashkda_contract", "flashkda_packed")


def draw(kind: str, c: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """The case's inputs as float32 NumPy arrays holding exactly the values every implementation sees."""
    B, T, Hk, K, V = c["B"], c["T"], c["H"], c["K"], c["V"]
    H = c.get("HV", Hk)  # heads of v, the gates and the state
    ks = iter(jax.random.split(jax.random.key(seed), 16))
    dtype = jnp.float32 if c.get("dtype") == "float32" else BF16

    def normal(shape, scale=1.0):
        return scale * jax.random.normal(next(ks), shape, F32)

    q = normal((B, T, Hk, K))
    k = normal((B, T, Hk, K))
    if not c.get("l2norm"):
        k = ref.l2norm(k)  # the delta rule wants unit-norm keys unless the op normalises them
    x = dict(q=q.astype(dtype), k=k.astype(dtype), v=normal((B, T, H, V)).astype(dtype))
    gate = c["gate"]
    if gate in ("lower_bound", "softplus"):
        raw = BF16 if c.get("flashkda") else F32  # FlashKDA takes the raw gate and beta logits in bf16
        x["g"] = normal((B, T, H, K)).astype(raw)  # raw gate input
        x["A_log"] = jnp.log(jax.random.uniform(next(ks), (H,), F32, 1.0, 16.0))
        x["dt_bias"] = normal((H * K,), 0.1)
        if kind == "kda":
            x["beta"] = normal((B, T, H)).astype(raw)  # beta logits
        else:  # GDN-2's erase and write logits
            x["b"], x["w"] = normal((B, T, H, K)).astype(raw), normal((B, T, H, V)).astype(raw)
    else:
        if gate == "extreme":
            u = jax.random.uniform(next(ks), (B, T, H, K))
            strong = -jax.random.uniform(next(ks), (B, T, H, K), F32, 10.0, 60.0)
            none = -jax.random.uniform(next(ks), (B, T, H, K), F32, 0.0, 1e-4)
            g = jnp.where(u < 0.15, strong, jnp.where(u < 0.5, none, jax.nn.log_sigmoid(normal((B, T, H, K)))))
            g = g.at[:, 128:256].set(-30.0)  # two whole chunks of huge decay
            g = g.at[:, 300:303].set(-1000.0)  # decays that underflow to zero
            g = g.at[:, 400:528].set(0.0)  # two chunks with no decay at all
            sharp = 6.0
        else:
            g = jax.nn.log_sigmoid(normal((B, T, H, K)))
            sharp = 1.0
        x["g"] = g
        if kind == "kda":
            x["beta"] = jax.nn.sigmoid(normal((B, T, H), sharp))
        else:
            x["b"] = jax.nn.sigmoid(normal((B, T, H, K), sharp))
            x["w"] = jax.nn.sigmoid(normal((B, T, H, V), sharp))
    n = len(c["cu"]) - 1 if "cu" in c else B
    if c.get("h0"):  # packed: the state entering the row's first document only
        x["h0"] = normal((B if "cu" in c else n, H, K, V), 0.5)
    x["do"] = normal((B, T, H, V))
    x["dht"] = normal((n, H, K, V))
    return {name: np.asarray(jnp.asarray(val).astype(F32)) for name, val in x.items()}


def segment_ids(c: dict) -> np.ndarray | None:
    if "cu" not in c:
        return None
    cu = np.asarray(c["cu"])
    return np.searchsorted(cu[1:], np.arange(c["T"]), side="right").astype(np.int32)[None]


def gate_fn(kind: str, gate: str):
    """``(g, gates)`` from the gate inputs: the activated gates for the raw-gate cases (FLA's formulas
    for ``g``; sigmoids of the beta, or GDN-2's erase and write, logits)."""

    def fn(x: dict) -> tuple[jax.Array, dict]:
        if gate not in ("lower_bound", "softplus"):
            gates = dict(beta=x["beta"]) if kind == "kda" else dict(b=x["b"], w=x["w"])
            return x["g"], gates
        H, K = x["g"].shape[-2:]
        a = jnp.exp(x["A_log"].astype(F32))[:, None]
        z = x["g"].astype(F32) + x["dt_bias"].astype(F32).reshape(H, K)
        g = LOWER_BOUND * jax.nn.sigmoid(a * z) if gate == "lower_bound" else -a * jax.nn.softplus(z)
        names = ("beta",) if kind == "kda" else ("b", "w")
        return g, {n: jax.nn.sigmoid(x[n].astype(F32)) for n in names}

    return fn


def diff_names(kind: str, c: dict) -> list[str]:
    names = ["q", "k", "v", "g"]
    names += ["beta"] if kind == "kda" else ["b", "w"]
    if c["gate"] in ("lower_bound", "softplus"):
        names += ["A_log", "dt_bias"]
    if c.get("h0"):
        names.append("h0")
    return names


def run_jla(kind: str, c: dict, x: dict[str, np.ndarray], impl: str) -> dict[str, np.ndarray]:
    """``o``, ``final`` and the gradients (``d<name>``) of one implementation."""
    names = diff_names(kind, c)
    upcast = impl == "reference"
    io_dtype = F32 if upcast or c.get("dtype") == "float32" else BF16
    seg = segment_ids(c)
    seg = None if seg is None else jnp.asarray(seg)
    gfn = gate_fn(kind, c["gate"])
    scale = c["K"] ** -0.5
    groups = c.get("HV", c["H"]) // c["H"]

    def f(*primals):
        xs = dict(zip(names, primals))
        g, gates = gfn(xs)
        h0 = xs.get("h0")
        q, k = (jnp.repeat(xs[n], groups, axis=2) if groups > 1 else xs[n] for n in ("q", "k"))
        o, final = ops.linear_attention(kind, q, k, xs["v"], impl=impl, g=g, **gates, scale=scale,
                                        initial_state=h0, output_final_state=True, segment_ids=seg,
                                        use_qk_l2norm=bool(c.get("l2norm")))
        return o, final

    primals = [jnp.asarray(x[n], io_dtype if n in ("q", "k", "v") else F32) for n in names]
    do = jnp.asarray(x["do"])
    dht = jnp.asarray(x["dht"][-1:] if "cu" in c else x["dht"])

    @jax.jit
    def fwd_bwd(*p):
        (o, final), vjp = jax.vjp(f, *p)
        return o, final, vjp((do.astype(o.dtype), dht))

    o, final, grads = fwd_bwd(*primals)
    out = dict(o=np.asarray(o.astype(F32)), final=np.asarray(final))
    out.update({f"d{n}": np.asarray(gr.astype(F32)) for n, gr in zip(names, grads)})
    return out


def oracle64(kind: str, c: dict, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Forward ``o`` and ``final`` in float64, token by token, written from the equations."""
    f64 = np.float64
    q, k, v = (x[n].astype(f64) for n in ("q", "k", "v"))
    groups = c.get("HV", c["H"]) // c["H"]
    q, k = np.repeat(q, groups, axis=2), np.repeat(k, groups, axis=2)  # value head h reads key head h // groups
    if c["gate"] in ("lower_bound", "softplus"):
        H, K = q.shape[2], q.shape[3]
        z = x["g"].astype(f64) + x["dt_bias"].astype(f64).reshape(H, K)
        a = np.exp(x["A_log"].astype(f64))[:, None]
        g = LOWER_BOUND / (1 + np.exp(-a * z)) if c["gate"] == "lower_bound" else -a * np.logaddexp(0.0, z)
        if kind == "kda":
            beta = 1 / (1 + np.exp(-x["beta"].astype(f64)))
            erase, write = beta[..., None], beta[..., None]
        else:
            erase, write = (1 / (1 + np.exp(-x[n].astype(f64))) for n in ("b", "w"))
    else:
        g = x["g"].astype(f64)
        if kind == "kda":
            erase = write = x["beta"].astype(f64)[..., None]
        else:
            erase, write = x["b"].astype(f64), x["w"].astype(f64)
    if c.get("l2norm"):
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    B, T, H, K = q.shape
    scale = K**-0.5
    seg = segment_ids(c)
    o = np.zeros(v.shape, f64)
    finals = []
    for b in range(B):
        state = x["h0"][b].astype(f64) if c.get("h0") else np.zeros((H, K, v.shape[-1]), f64)
        for t in range(T):
            if seg is not None and t > 0 and seg[b, t] != seg[b, t - 1]:
                state = np.zeros_like(state)
            state = state * np.exp(g[b, t])[:, :, None]  # S' = D_t S_{t-1}
            e = erase[b, t] * k[b, t]  # read (erase) direction b_t * k_t
            recalled = np.einsum("hk,hkv->hv", e, state)  # S'^T e_t
            state = state + k[b, t][:, :, None] * (write[b, t] * v[b, t] - recalled)[:, None, :]
            o[b, t] = scale * np.einsum("hk,hkv->hv", q[b, t], state)
        finals.append(state)
    return dict(o=o, final=np.stack(finals))


def errors(x: np.ndarray, want: np.ndarray) -> dict[str, float]:
    x, want = np.asarray(x, np.float64), np.asarray(want, np.float64)
    scale = max(float(np.abs(want).max()), 1e-30)
    norm = max(float(np.linalg.norm(want)), 1e-30)
    out = dict(max=float(np.abs(x - want).max()) / scale, rms=float(np.linalg.norm(x - want)) / norm)
    if not np.isfinite(x).all():
        out["nonfinite"] = int((~np.isfinite(x)).sum())
    return out


def run_fla(torch_python: str, kind: str, name: str, c: dict, x: dict[str, np.ndarray]) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_kda_diff.py")
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(inp, **x)
        meta = dict(c, names=diff_names(kind, c), case=name)
        proc = subprocess.run([torch_python, script, inp, outp, json.dumps(meta)], capture_output=True, text=True,
                              timeout=1800)
        if proc.returncode != 0 or not os.path.exists(outp):
            return dict(error=harness.error_summary(proc.stderr or proc.stdout, 1500))
        info = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        with np.load(outp) as data:
            out = {key: data[key] for key in data.files}
        out["_info"] = json.loads(info[-1]) if info else {}
        return out


def diff_case(kind: str, name: str, c: dict, torch_python: str | None) -> dict:
    x = draw(kind, c)
    t0 = time.time()
    impls = ["reference", "chunked"] + ([] if c.get("dtype") == "float32" else ["mosaic_gpu"])
    res = {impl: run_jla(kind, c, x, impl) for impl in impls}
    want = res["reference"]
    o64 = oracle64(kind, c, x)  # with cu, final is the state after the row's last token (as in jla)
    report = dict(case=name, config={k: v for k, v in c.items()}, tensors={})
    report["tensors"]["o"] = {"reference_vs_oracle64": errors(want["o"], o64["o"])}
    report["tensors"]["final"] = {"reference_vs_oracle64": errors(want["final"], o64["final"])}
    fla = None
    if kind == "kda" and torch_python:
        fla = run_fla(torch_python, kind, name, c, x)
        if "error" in fla:
            report["fla_error"] = fla["error"]
            fla = None
        else:
            report["fla_info"] = fla.pop("_info")
    if fla is not None and "o_flashkda" in fla:  # FlashKDA forward (inference only, no state)
        row = report["tensors"]["o"]
        row["flashkda"] = errors(fla["o_flashkda"], want["o"])
        if "mosaic_gpu" in res:
            diff = errors(res["mosaic_gpu"]["o"], fla["o_flashkda"])
            diff["max"] = float(np.abs(res["mosaic_gpu"]["o"] - fla["o_flashkda"]).max()) / max(
                float(np.abs(want["o"]).max()), 1e-30)
            row["mosaic_gpu_vs_flashkda"] = diff
    for tensor in want:
        row = report["tensors"].setdefault(tensor, {})
        for impl in impls[1:]:
            row[impl] = errors(res[impl][tensor], want[tensor])
        if fla is not None and tensor in fla:
            row["fla"] = errors(fla[tensor], want[tensor])
            if "mosaic_gpu" in res:
                diff = errors(res["mosaic_gpu"][tensor], fla[tensor])
                scale = max(float(np.abs(want[tensor]).max()), 1e-30)
                diff["max"] = float(np.abs(res["mosaic_gpu"][tensor] - fla[tensor]).max()) / scale
                row["mosaic_gpu_vs_fla"] = diff
    report["wall_s"] = round(time.time() - t0, 1)
    return report


def causality_case(kind: str, name: str, c: dict) -> dict:
    """Prefix outputs and prefix-loss gradients with the gates after ``P`` edited."""
    base = dict(c, gate="plain")
    x = draw(kind, base)
    P = c["P"]
    x_edit = dict(x)
    g = x["g"].copy()
    g[:, P:] = c["edit"]
    x_edit["g"] = g
    for xx in (x, x_edit):
        xx["do"] = xx["do"].copy()
        xx["do"][:, P:] = 0.0
        xx["dht"] = np.zeros_like(xx["dht"])
    out = dict(case=name, config=c, impls={})
    for impl in ("reference", "chunked", "mosaic_gpu"):
        a, b = run_jla(kind, base, x, impl), run_jla(kind, base, x_edit, impl)
        row = {}
        for tensor in a:
            if tensor == "final":
                continue
            ta, tb = (a[tensor][:, :P], b[tensor][:, :P]) if tensor in ("o", "dq", "dk", "dv", "dg", "dbeta",
                                                                          "db", "dw") else (a[tensor], b[tensor])
            row[tensor] = dict(bitwise=bool(np.array_equal(ta, tb)),
                               max=float(np.abs(ta - tb).max()) / max(float(np.abs(ta).max()), 1e-30))
        out["impls"][impl] = row
    return out


def summarize(results: dict) -> None:
    """One line per case and tensor: ``impl=max/rms`` (relative errors, see the module docstring).

    A merged file (``runs``: the cases of each baseline environment) prints each run in turn."""
    if "runs" in results:
        for run_name, cases in results["runs"].items():
            print(f"== run {run_name}")
            summarize(dict(cases=cases))
        return
    for entry in results["cases"]:
        if "error" in entry:
            print(f"{entry['case']}: ERROR {entry['error'][:300]}")
            continue
        if "impls" in entry:  # causality
            for impl, row in entry["impls"].items():
                cells = " ".join(f"{t}={'bitwise' if r['bitwise'] else format(r['max'], '.1e')}"
                                 for t, r in row.items())
                print(f"{entry['case']} {impl}: {cells}")
            continue
        for tensor, row in entry["tensors"].items():
            cells = " ".join(f"{impl}={r['max']:.1e}/{r['rms']:.1e}" + ("(NONFINITE)" if "nonfinite" in r else "")
                             for impl, r in row.items())
            print(f"{entry['case']} {tensor}: {cells}")
        if "fla_error" in entry:
            print(f"{entry['case']}: FLA error {entry['fla_error'][:300]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("kda", "gdn2"), default=None)
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + FLA (KDA)")
    parser.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    parser.add_argument("--out", default=None)
    parser.add_argument("--summarize", default=None, help="print the tables of a results file and exit")
    args = parser.parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            summarize(json.load(f))
        return
    if args.kind is None:
        parser.error("--kind is required")
    names = [n for n in list(CASES) + list(CAUSALITY) if args.kind == "kda" or n not in KDA_ONLY]
    if args.cases:
        names = [n for n in args.cases.split(",") if n]
    results = dict(kind=args.kind, environment=harness.environment(), cases=[])
    for name in names:
        try:
            if name in CAUSALITY:
                entry = causality_case(args.kind, name, CAUSALITY[name])
            else:
                entry = diff_case(args.kind, name, CASES[name], args.torch_python)
        except Exception as err:  # noqa: BLE001  (record and continue)
            entry = dict(case=name, error=f"{type(err).__name__}: {str(err)[:800]}")
        results["cases"].append(entry)
        print(json.dumps(entry), flush=True)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                   f"{args.kind}_diff_{time.strftime('%Y%m%d_%H%M')}.json")
    harness.write_json(out, results)
    summarize(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
