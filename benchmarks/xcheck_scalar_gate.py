# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Differential check of Gated DeltaNet, DeltaNet, simple GLA and Mamba-2 SSD against upstream.

    python benchmarks/xcheck_scalar_gate.py gen    --dir DIR [--cases a,b]
    python benchmarks/xcheck_scalar_gate.py ours   --dir DIR [--cases a,b] [--impls reference,chunked,mosaic_gpu]
    (torch env) python benchmarks/xcheck_scalar_gate_torch.py --dir DIR --cases a,b --impls oracle,fla,...
    python benchmarks/xcheck_scalar_gate.py report --dir DIR [--out results.json]

``gen`` writes identical random inputs (bf16-representable values when the case is bf16)
and output cotangents for every case; ``ours`` runs ``jla.ops.linear_attention`` with each
implementation and writes the outputs and the gradients of every input; the torch script
writes the same for a float64 token-loop oracle, FLA, FlashQLA and the official Mamba-2
kernel. ``report`` prints, per case and tensor, ``max |x - truth| / max |truth|`` with
``truth`` the float64 oracle (or, for long cases where the oracle runs forward only, our
float32 ``chunked`` op at HIGHEST precision, marked ``f32*``), and the same error of ours
against each upstream op.

Mamba-2 cases run our ``simple_gla`` through the SSD mapping ``q = C``, ``k = B``,
``v = dt x``, ``g = dt A``, ``scale = 1`` and compare the gradients of ``x, dt, A, B, C``
and the initial state with ``mamba_chunk_scan_combined``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zlib

import numpy as np

CASES: dict[str, dict] = {}


def case(name: str, **kw) -> None:
    CASES[name] = kw


# kind, B, T, Hk, H, K, V, docs, h0, gates, dtype; long cases get a forward-only oracle.
for kind, tag in (("gated_deltanet", "gdn"), ("deltanet", "dn"), ("simple_gla", "sgla")):
    case(f"{tag}_dense_t1000", kind=kind, B=2, T=1000, Hk=4, H=4, K=128, V=128, docs=None, h0=True)
    case(f"{tag}_docs_rand", kind=kind, B=2, T=1536, Hk=4, H=4, K=128, V=128, docs="random", h0=True)
    case(f"{tag}_k64_docs", kind=kind, B=2, T=777, Hk=4, H=4, K=64, V=64, docs="random", h0=True)
    case(f"{tag}_gqa_docs", kind=kind, B=1, T=1100, Hk=2, H=8, K=128, V=128 if kind != "simple_gla" else 64,
         docs="random", h0=True)
    case(f"{tag}_f32", kind=kind, B=1, T=700, Hk=2, H=2, K=64, V=64, docs="random", h0=True, dtype="float32")
    case(f"{tag}_long_docs", kind=kind, B=1, T=16384, Hk=16, H=16, K=128, V=128, docs="random", h0=True,
         oracle_fwd_only=True)
    case(f"{tag}_long_dense_odd", kind=kind, B=1, T=8191, Hk=8, H=8, K=128, V=128, docs=None, h0=True,
         oracle_fwd_only=True)
    if kind != "deltanet":
        case(f"{tag}_extreme", kind=kind, B=1, T=1024, Hk=4, H=4, K=128, V=128, docs="random", h0=True,
             gates="extreme")
    # B * H * V / 64 > 32: no sequence split (one CTA per (b, h, 64 value columns) walks the whole row).
    case(f"{tag}_big_bh_docs", kind=kind, B=4, T=700, Hk=8, H=8, K=128, V=128, docs="random", h0=True)
case("sgla_shared_bc", kind="simple_gla", B=2, T=1000, Hk=1, H=8, K=128, V=64, docs="random", h0=True)
# beta in (0, 2): FLA's allow_neg_eigval range (the transition I - beta k k^T gets negative eigenvalues).
case("gdn_beta2", kind="gated_deltanet", B=1, T=1000, Hk=4, H=4, K=128, V=128, docs="random", h0=True,
     beta_scale=2.0)
case("mamba2_big_bh_docs", kind="mamba2", B=2, T=700, G=1, H=32, N=128, P=64, docs="random", h0=True)
# q/k L2-normalized inside the op (FLA use_qk_l2norm_in_kernel, ours use_qk_l2norm), raw inputs.
case("gdn_l2norm_docs", kind="gated_deltanet", B=2, T=1000, Hk=2, H=4, K=128, V=128, docs="random", h0=True,
     l2norm=True)
case("dn_l2norm_docs", kind="deltanet", B=2, T=1000, Hk=4, H=4, K=128, V=128, docs="random", h0=True, l2norm=True)
case("mamba2_docs", kind="mamba2", B=2, T=1000, G=1, H=8, N=128, P=64, docs="random", h0=True)
case("mamba2_dense", kind="mamba2", B=2, T=1024, G=1, H=8, N=128, P=64, docs=None, h0=True)
case("mamba2_extreme", kind="mamba2", B=1, T=1000, G=1, H=8, N=128, P=64, docs="random", h0=True,
     gates="extreme")
case("mamba2_long_docs", kind="mamba2", B=1, T=16384, G=1, H=16, N=128, P=64, docs="random", h0=True,
     oracle_fwd_only=True)
# FLA's 1.3B Gated DeltaNet head (K = 256, expand_v = 2: V = 512) and V = 256 (scalar_gate_kb).
case("gdn_hd256_docs", kind="gated_deltanet", B=2, T=1000, Hk=2, H=2, K=256, V=512, docs="random", h0=True)
case("gdn_hd256_gqa_dense_odd", kind="gated_deltanet", B=1, T=777, Hk=1, H=2, K=256, V=512, docs=None, h0=True)
case("gdn_hd256_v256_docs", kind="gated_deltanet", B=1, T=1100, Hk=2, H=2, K=256, V=256, docs="random", h0=True)
case("gdn_hd256_extreme", kind="gated_deltanet", B=1, T=1024, Hk=2, H=2, K=256, V=512, docs="random", h0=True,
     gates="extreme")
case("dn_hd256_docs", kind="deltanet", B=1, T=1000, Hk=2, H=2, K=256, V=512, docs="random", h0=True)
case("gdn_hd256_long_docs", kind="gated_deltanet", B=1, T=8192, Hk=6, H=6, K=256, V=512, docs="random", h0=True,
     oracle_fwd_only=True)


# --------------------------------------------------------------------------------------- inputs
def round_to(x: np.ndarray, dtype: str) -> np.ndarray:
    import ml_dtypes

    x = x.astype(np.float32)
    return x.astype(ml_dtypes.bfloat16).astype(np.float32) if dtype == "bfloat16" else x


def doc_segments(rng, batch: int, seq_len: int) -> np.ndarray:
    """Random document lengths: 1-token documents, chunk-aligned starts, and long documents."""
    seg = np.zeros((batch, seq_len), np.int32)
    for b in range(batch):
        t, idx = 0, 0
        while t < seq_len:
            r = rng.random()
            if r < 0.15:
                n = 1
            elif r < 0.35:
                n = 64 * int(rng.integers(1, 6))
            elif r < 0.5:
                n = 64 - (t % 64) or 64  # end exactly on a chunk boundary
            else:
                n = int(rng.integers(2, max(3, seq_len // 3)))
            seg[b, t:t + n] = idx
            t += n
            idx += 1
    return seg


def logsigmoid(x):
    return -np.logaddexp(0.0, -x)


def gen_case(name: str, spec: dict) -> dict:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    kind, batch, seq_len = spec["kind"], spec["B"], spec["T"]
    dtype = spec.get("dtype", "bfloat16")
    extreme = spec.get("gates") == "extreme"
    out = {}
    if kind == "mamba2":
        g_, heads, n, p = spec["G"], spec["H"], spec["N"], spec["P"]
        out["x"] = round_to(rng.standard_normal((batch, seq_len, heads, p)), dtype)
        out["B"] = round_to(rng.standard_normal((batch, seq_len, g_, n)) * 0.5, dtype)
        out["C"] = round_to(rng.standard_normal((batch, seq_len, g_, n)) * 0.5, dtype)
        dt = np.logaddexp(0.0, rng.standard_normal((batch, seq_len, heads)) - 3.0)
        a = -rng.uniform(0.5, 4.0, heads)
        if extreme:  # some tokens forget everything (dt A ~ -100), some keep everything (dt ~ 0)
            mask = rng.random((batch, seq_len, heads))
            dt = np.where(mask < 0.1, 25.0, np.where(mask < 0.3, 1e-6, dt))
            a = -rng.uniform(1.0, 16.0, heads)
        out["dt"] = dt.astype(np.float32)
        out["A"] = a.astype(np.float32)
        if spec.get("h0"):
            out["h0"] = (rng.standard_normal((batch, heads, p, n)) * 0.5).astype(np.float32)
        out["do"] = round_to(rng.standard_normal((batch, seq_len, heads, p)), dtype)
        out["dht"] = (rng.standard_normal((batch, heads, p, n)) * 0.1).astype(np.float32)
        scale = 1.0
    else:
        hk, heads, kd, vd = spec["Hk"], spec["H"], spec["K"], spec["V"]
        q = rng.standard_normal((batch, seq_len, hk, kd))
        k = rng.standard_normal((batch, seq_len, hk, kd))
        if kind != "simple_gla" and spec.get("l2norm"):  # raw q, k; the op normalizes them
            scale = kd ** -0.5
        elif kind != "simple_gla":
            k = k / np.linalg.norm(k, axis=-1, keepdims=True)
            q = q / np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(kd) * 0.5
            scale = kd ** -0.5
        else:
            q, k = q * 0.25, k * 0.25
            scale = 1.0
        out["q"] = round_to(q, dtype)
        out["k"] = round_to(k, dtype)
        out["v"] = round_to(rng.standard_normal((batch, seq_len, heads, vd)), dtype)
        if kind != "deltanet":
            g = logsigmoid(rng.normal(3.0, 1.5, (batch, seq_len, heads)))
            if extreme:
                mask = rng.random((batch, seq_len, heads))
                g = np.where(mask < 0.1, -80.0, np.where(mask < 0.3, -1e-7, g))
            out["g"] = g.astype(np.float32)
        if kind != "simple_gla":
            beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((batch, seq_len, heads))))
            if extreme:
                mask = rng.random((batch, seq_len, heads))
                beta = np.where(mask < 0.1, 0.0, np.where(mask < 0.2, 1.0, beta))
            out["beta"] = (beta * spec.get("beta_scale", 1.0)).astype(np.float32)
        if spec.get("h0"):
            out["h0"] = (rng.standard_normal((batch, heads, kd, vd)) * 0.5).astype(np.float32)
        out["do"] = round_to(rng.standard_normal((batch, seq_len, heads, vd)), dtype)
        out["dht"] = (rng.standard_normal((batch, heads, kd, vd)) * 0.1).astype(np.float32)
    if spec.get("docs") == "random":
        out["seg"] = doc_segments(rng, batch, seq_len)
    meta = dict(spec, dtype=dtype, scale=scale, name=name)
    out["meta"] = np.array(json.dumps(meta))
    return out


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


# ----------------------------------------------------------------------------------------- ours
def run_ours(case_in: dict, impl: str) -> dict:
    import jax
    import jax.numpy as jnp

    from jla import ops

    meta = json.loads(str(case_in["meta"]))
    kind = meta["kind"]
    dtype = jnp.float32 if impl in ("reference", "chunked_f32") else jnp.dtype(meta["dtype"])
    # mosaic_rep_bf16 / mosaic_rep_f32: the kernels with grouped q/k (B/C) repeated before the op, as the
    # layers do, the repeat done in the input dtype or through float32 (so its transpose sums in float32).
    op_impl = {"chunked_f32": "chunked", "mosaic_rep_bf16": "mosaic_gpu", "mosaic_rep_f32": "mosaic_gpu"}.get(
        impl, impl)

    def repeat(x, rep):
        if impl == "mosaic_rep_f32":
            return jnp.repeat(x.astype(jnp.float32), rep, axis=2).astype(x.dtype)
        return jnp.repeat(x, rep, axis=2)

    seg = jnp.asarray(case_in["seg"]) if "seg" in case_in else None
    opts = dict(output_final_state=True)
    if meta.get("l2norm"):
        opts["use_qk_l2norm"] = True
    if seg is not None:
        opts["segment_ids"] = seg
    if impl == "chunked_f32":
        opts["precision"] = jax.lax.Precision.HIGHEST
    f32 = jnp.float32

    if kind == "mamba2":
        names = ["x", "dt", "A", "B", "C"] + (["h0"] if "h0" in case_in else [])

        def fn(x, dt, a, bm, cm, h0=None):
            v = (dt[..., None] * x.astype(f32)).astype(dtype)
            g = dt * a
            state = None if h0 is None else jnp.swapaxes(h0, -1, -2)
            if impl != "mosaic_gpu" and cm.shape[2] != v.shape[2]:  # only the kernels take grouped B/C
                rep = v.shape[2] // cm.shape[2]
                cm, bm = repeat(cm, rep), repeat(bm, rep)
            o, ht = ops.linear_attention("simple_gla", cm.astype(dtype), bm.astype(dtype), v, g=g, impl=op_impl,
                                         scale=1.0, initial_state=state, **opts)
            return o, jnp.swapaxes(ht, -1, -2)

        leaves = [jnp.asarray(case_in[n], dtype if n in ("x", "B", "C") else f32) for n in names]
    else:
        names = ["q", "k", "v"] + [n for n in ("g", "beta", "h0") if n in case_in]

        def fn(q, k, v, *rest):
            kw = dict(zip(names[3:], rest))
            h0 = kw.pop("h0", None)
            heads = v.shape[2]
            if impl != "mosaic_gpu" and q.shape[2] != heads:
                rep = heads // q.shape[2]
                q, k = repeat(q, rep), repeat(k, rep)
            return ops.linear_attention(kind, q, k, v, impl=op_impl, scale=meta["scale"], initial_state=h0,
                                        **kw, **opts)

        leaves = [jnp.asarray(case_in[n], dtype if n in ("q", "k", "v") else f32) for n in names]
    (o, ht), vjp = jax.vjp(fn, *leaves)
    do = jnp.asarray(case_in["do"], o.dtype)
    dht = jnp.asarray(case_in["dht"], ht.dtype)
    grads = vjp((do, dht))
    res = {"o": np.asarray(o.astype(f32)), "ht": np.asarray(ht.astype(f32))}
    for n, gr in zip(names, grads):
        res["d" + n] = np.asarray(gr.astype(f32))
    return res


# --------------------------------------------------------------------------------------- report
def compare(x: np.ndarray, ref: np.ndarray) -> dict:
    x = x.astype(np.float64)
    ref = ref.astype(np.float64)
    if x.shape != ref.shape:
        return dict(error=f"shape {x.shape} vs {ref.shape}")
    if not np.all(np.isfinite(x)):
        return dict(error="non-finite")
    diff = np.abs(x - ref)
    scale = float(np.max(np.abs(ref))) or 1.0
    rms = float(np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(ref ** 2)) or 1.0))
    return dict(max_abs=float(diff.max()), rel=float(diff.max()) / scale, rms_rel=rms)


def report(directory: str, names: list[str], out: str | None) -> dict:
    ours_impls = ["reference", "chunked_f32", "chunked", "mosaic_gpu", "mosaic_rep_bf16", "mosaic_rep_f32"]
    up_impls = ["fla", "fla_full", "flashqla", "flashqla_main", "mamba2", "fla_ssd"]
    rows = {}
    for name in names:
        paths = {impl: os.path.join(directory, f"{name}.{impl}.npz") for impl in
                 ["oracle"] + [f"ours_{i}" for i in ours_impls] + up_impls}
        res = {impl: load(p) for impl, p in paths.items() if os.path.exists(p)}
        errors = {impl: str(r["error"]) for impl, r in res.items() if "error" in r}
        res = {impl: r for impl, r in res.items() if "error" not in r}
        oracle = res.get("oracle")
        f32ref = res.get("ours_chunked_f32")
        row = dict(errors=errors, vs_truth={}, vs_upstream={})
        tensors = sorted({t for r in res.values() for t in r})
        for t in tensors:
            if oracle is not None and t in oracle:
                truth, label = oracle[t], "oracle_f64"
            elif f32ref is not None and t in f32ref:
                truth, label = f32ref[t], "f32*"
            else:
                continue
            for impl, r in res.items():
                if impl in ("oracle",) or t not in r or (label == "f32*" and impl == "ours_chunked_f32"):
                    continue
                row["vs_truth"].setdefault(impl, {})[t] = dict(compare(r[t], truth), truth=label)
            ours = res.get("ours_mosaic_gpu") or res.get("ours_chunked")
            if ours is not None and t in ours:
                for up in up_impls:
                    if up in res and t in res[up]:
                        row["vs_upstream"].setdefault(up, {})[t] = compare(ours[t], res[up][t])
        rows[name] = row
    print_report(rows)
    print_markdown(rows)
    if out:
        with open(out, "w") as f:
            json.dump(dict(cases={n: CASES.get(n) for n in names}, rows=rows), f, indent=1)
    return rows


def print_markdown(rows: dict) -> None:
    """One line per case: the worst tensor's error (max-normalized; rms-normalized in brackets) per implementation."""
    cols = ["ours_reference", "ours_chunked_f32", "ours_chunked", "ours_mosaic_gpu", "ours_mosaic_rep_bf16",
            "ours_mosaic_rep_f32", "fla", "fla_full", "flashqla_main", "flashqla", "mamba2", "fla_ssd"]
    print("| case | truth | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for name, row in rows.items():
        cells = []
        label = "+".join(sorted({v["truth"] for per in row["vs_truth"].values() for v in per.values()})) or "-"
        for impl in cols:
            per = row["vs_truth"].get(impl)
            if not per:
                cells.append("err" if impl in row["errors"] else "-")
                continue
            vals = [v for v in per.values() if "rel" in v]
            if len(vals) < len(per):
                cells.append("non-finite")
                continue
            worst = max(vals, key=lambda v: v["rel"])
            worst_t = next(t for t, v in per.items() if v is worst)
            cells.append(f"{worst['rel']:.1e} ({worst_t}) [{max(v['rms_rel'] for v in vals):.1e}]")
        print(f"| {name} | {label} | " + " | ".join(cells) + " |")


def print_report(rows: dict) -> None:
    for name, row in rows.items():
        print(f"== {name}")
        for impl, err in row["errors"].items():
            print(f"   {impl}: ERROR {err[:200]}")
        for impl, per in row["vs_truth"].items():
            cells = " ".join(f"{t}={v.get('rel', float('nan')):.1e}" if "rel" in v else f"{t}={v.get('error')}"
                             for t, v in per.items())
            label = next(iter(per.values())).get("truth")
            print(f"   {impl:18s} vs {label:10s} {cells}")
        for up, per in row["vs_upstream"].items():
            cells = " ".join(f"{t}={v.get('rel', float('nan')):.1e}" if "rel" in v else f"{t}={v.get('error')}"
                             for t, v in per.items())
            print(f"   ours vs {up:9s}        {cells}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gen", "ours", "report", "list"))
    parser.add_argument("--dir", required=True)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--impls", default="reference,chunked_f32,chunked,mosaic_gpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    names = args.cases.split(",") if args.cases else list(CASES)
    os.makedirs(args.dir, exist_ok=True)
    if args.mode == "list":
        print(",".join(names))
    elif args.mode == "gen":
        for name in names:
            np.savez(os.path.join(args.dir, f"{name}.in.npz"), **gen_case(name, CASES[name]))
        print(f"wrote {len(names)} cases to {args.dir}")
    elif args.mode == "ours":
        for name in names:
            case_in = load(os.path.join(args.dir, f"{name}.in.npz"))
            meta = json.loads(str(case_in["meta"]))
            for impl in args.impls.split(","):
                if impl.startswith("mosaic") and meta["dtype"] == "float32":
                    continue
                if impl == "reference" and meta.get("oracle_fwd_only"):
                    continue  # the token loop's autodiff residuals do not fit at 16K tokens
                t0 = time.time()
                path = os.path.join(args.dir, f"{name}.ours_{impl}.npz")
                try:
                    np.savez(path, **run_ours(case_in, impl))
                    print(json.dumps(dict(case=name, impl=impl, ok=True, s=round(time.time() - t0, 1))), flush=True)
                except Exception as err:  # noqa: BLE001 (record and continue)
                    np.savez(path, error=np.array(f"{type(err).__name__}: {err}"[:2000]))
                    print(json.dumps(dict(case=name, impl=impl, ok=False, error=f"{type(err).__name__}: "
                                          f"{str(err)[:300]}")), flush=True)
                    traceback.print_exc(file=sys.stderr)
    else:
        report(args.dir, names, args.out)


if __name__ == "__main__":
    main()
