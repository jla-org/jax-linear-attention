# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch side of the scalar-gate differential check (run in a torch environment).

    python xcheck_scalar_gate_torch.py --dir CASE_DIR --cases a,b --impls oracle,fla,flashqla,mamba2

Reads ``CASE_DIR/<case>.in.npz`` (written by ``xcheck_scalar_gate.py``) and writes
``CASE_DIR/<case>.<impl>.npz`` with the outputs and the gradients of every input for the
cotangents stored in the case:

* ``oracle``: a token-by-token float64 recurrence with torch autograd (written here,
  independent of FLA and of our code);
* ``fla``: FLA ``chunk_gated_delta_rule`` / ``chunk_delta_rule`` / ``chunk_simple_gla``
  (whatever FLA this environment has: Triton, or its default dispatch);
* ``flashqla``: ``flash_qla.chunk_gated_delta_rule`` (Gated DeltaNet, K = V = 128);
* ``mamba2``: ``mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined``;
* ``fla_ssd`` (Mamba-2 cases): FLA ``chunk_simple_gla`` through the SSD mapping, as our side runs it.

Packed documents: our ``segment_ids [B, T]`` become one ``cu_seqlens`` row of ``B * T``
tokens for FLA and FlashQLA (the initial state goes to each row's first document, the final
state is read at each row's last document) and ``seq_idx`` for Mamba-2. Grouped key heads
are passed natively where the op supports them (FLA and FlashQLA Gated DeltaNet: GVA) and
repeated through float32 otherwise (FLA DeltaNet and simple GLA; autograd sums the repeats in
float32, as our layers do).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.25)
DEV = "cuda"


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def meta_of(case: dict) -> dict:
    return json.loads(str(case["meta"]))


def docs_of(seg: np.ndarray) -> list[list[tuple[int, int]]]:
    """Per row, the ``(start, end)`` token ranges of its documents."""
    rows = []
    for row in seg:
        starts = [0] + [t for t in range(1, len(row)) if row[t] != row[t - 1]]
        ends = starts[1:] + [len(row)]
        rows.append(list(zip(starts, ends)))
    return rows


def repeat_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return x if x.shape[2] == heads else x.repeat_interleave(heads // x.shape[2], dim=2)


def repeat_heads_f32(x: torch.Tensor, heads: int) -> torch.Tensor:
    """Repeat grouped key heads through float32, so autograd sums the repeats in float32 (as our layers do)."""
    return x if x.shape[2] == heads else repeat_heads(x.float(), heads).to(x.dtype)


# ----------------------------------------------------------------------------------------- oracle
def oracle_recurrence(q, k, v, g, beta, h0, resets, scale: float, delta: bool):
    """float64 token loop. q, k [B, T, H, K] (already per value head), v [B, T, H, V]."""
    batch, seq_len, heads, _ = q.shape
    state = h0
    outs = []
    for t in range(seq_len):
        if resets is not None:
            keep = (~resets[:, t]).to(state.dtype)[:, None, None, None]
            state = state * keep
        if g is not None:
            state = state * torch.exp(g[:, t])[:, :, None, None]
        k_t = k[:, t]
        if delta:
            recalled = torch.einsum("bhk,bhkv->bhv", k_t, state)
            u = beta[:, t][:, :, None] * (v[:, t] - recalled)
        else:
            u = v[:, t]
        state = state + k_t[:, :, :, None] * u[:, :, None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))
    return torch.stack(outs, dim=1), state


def run_oracle(case: dict, meta: dict) -> dict:
    """float64 oracle; ``oracle_fwd_only`` (long cases) skips autograd so no per-step state is kept."""
    with torch.set_grad_enabled(not meta.get("oracle_fwd_only", False)):
        return _run_oracle(case, meta)


def _run_oracle(case: dict, meta: dict) -> dict:
    f64 = torch.float64
    if meta["kind"] == "mamba2":
        rg = torch.is_grad_enabled()
        x, dt, a, bm, cm = (torch.tensor(case[n], device=DEV, dtype=f64, requires_grad=rg)
                            for n in ("x", "dt", "A", "B", "C"))
        heads = x.shape[2]
        h0 = torch.tensor(case["h0"], device=DEV, dtype=f64, requires_grad=rg) if "h0" in case else None
        q = repeat_heads(cm, heads)
        k = repeat_heads(bm, heads)
        v = dt[..., None] * x
        g = dt * a
        state0 = h0.transpose(-1, -2) if h0 is not None else torch.zeros(
            x.shape[0], heads, bm.shape[-1], x.shape[-1], device=DEV, dtype=f64)
        resets = seg_resets(case)
        o, ht = oracle_recurrence(q, k, v, g, None, state0, resets, 1.0, False)
        ht = ht.transpose(-1, -2)
        leaves = dict(x=x, dt=dt, A=a, B=bm, C=cm)
    else:
        rg = torch.is_grad_enabled()
        q, k, v = (torch.tensor(case[n], device=DEV, dtype=f64, requires_grad=rg) for n in ("q", "k", "v"))
        heads = v.shape[2]
        g = torch.tensor(case["g"], device=DEV, dtype=f64, requires_grad=rg) if "g" in case else None
        beta = torch.tensor(case["beta"], device=DEV, dtype=f64, requires_grad=rg) if "beta" in case else None
        h0 = torch.tensor(case["h0"], device=DEV, dtype=f64, requires_grad=rg) if "h0" in case else None
        state0 = h0 if h0 is not None else torch.zeros(v.shape[0], heads, q.shape[-1], v.shape[-1],
                                                       device=DEV, dtype=f64)
        resets = seg_resets(case)
        qn, kn = q, k
        if meta.get("l2norm"):  # FLA l2norm: x / sqrt(sum x^2 + 1e-6)
            qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        o, ht = oracle_recurrence(repeat_heads(qn, heads), repeat_heads(kn, heads), v, g, beta, state0, resets,
                                  meta["scale"], meta["kind"] != "simple_gla")
        leaves = dict(q=q, k=k, v=v)
        if g is not None:
            leaves["g"] = g
        if beta is not None:
            leaves["beta"] = beta
    if h0 is not None:
        leaves["h0"] = h0
    return backward(o, ht, leaves, case, grads=not meta.get("oracle_fwd_only", False))


def seg_resets(case: dict):
    if "seg" not in case:
        return None
    seg = torch.tensor(case["seg"], device=DEV)
    change = seg[:, 1:] != seg[:, :-1]
    return torch.cat([torch.zeros_like(change[:, :1]), change], dim=1)


def backward(o, ht, leaves: dict, case: dict, grads: bool = True) -> dict:
    if not grads:
        res = {"o": o.detach().float().cpu().numpy()}
        if ht is not None:
            res["ht"] = ht.detach().float().cpu().numpy()
        return res
    do = torch.tensor(case["do"], device=DEV, dtype=o.dtype)
    dht = torch.tensor(case["dht"], device=DEV, dtype=torch.float32)
    loss_terms = [(o, do)]
    if ht is not None:
        loss_terms.append((ht, dht.to(ht.dtype)))
    outs = [t for t, _ in loss_terms]
    cots = [c for _, c in loss_terms]
    names = list(leaves)
    grads = torch.autograd.grad(outs, [leaves[n] for n in names], grad_outputs=cots, allow_unused=True)
    res = {"o": o.detach().float().cpu().numpy()}
    if ht is not None:
        res["ht"] = ht.detach().float().cpu().numpy()
    for n, gr in zip(names, grads):
        res["d" + n] = (np.zeros(leaves[n].shape, np.float32) if gr is None else gr.detach().float().cpu().numpy())
    return res


# -------------------------------------------------------------------------------- FLA / FlashQLA
def run_fla_like(case: dict, meta: dict, impl: str) -> dict:
    kind = meta["kind"]
    dtype = getattr(torch, meta["dtype"])
    q, k, v = (torch.tensor(case[n], device=DEV, dtype=dtype, requires_grad=True) for n in ("q", "k", "v"))
    batch, seq_len, heads, value_dim = v.shape
    key_dim = q.shape[-1]
    g = torch.tensor(case["g"], device=DEV, dtype=torch.float32, requires_grad=True) if "g" in case else None
    beta = torch.tensor(case["beta"], device=DEV, dtype=torch.float32, requires_grad=True) if "beta" in case else None
    h0 = torch.tensor(case["h0"], device=DEV, dtype=torch.float32, requires_grad=True) if "h0" in case else None
    seg = case.get("seg")
    leaves = dict(q=q, k=k, v=v)
    if g is not None:
        leaves["g"] = g
    if beta is not None:
        leaves["beta"] = beta
    if h0 is not None:
        leaves["h0"] = h0

    def flat(x):
        return x.reshape(1, batch * seq_len, *x.shape[2:])

    if seg is not None:
        docs = docs_of(seg)
        starts = [b * seq_len + s for b, row in enumerate(docs) for s, _ in row]
        cu = torch.tensor(starts + [batch * seq_len], device=DEV, dtype=torch.long)
        first = [i for i, (b, row) in enumerate([(b, r) for b, rows in enumerate(docs) for r in rows])
                 if row[0] == 0]
        last = [i for i, (b, row) in enumerate([(b, r) for b, rows in enumerate(docs) for r in rows])
                if row[1] == seq_len]
        n_docs = len(starts)
        if h0 is not None:
            state0 = torch.zeros(n_docs, heads, key_dim, value_dim, device=DEV, dtype=torch.float32)
            state0 = state0.index_copy(0, torch.tensor(first, device=DEV), h0)
        else:
            state0 = None
        args = [flat(q), flat(k), flat(v)]
        kw = dict(cu_seqlens=cu if impl == "fla" else cu.int())
        gates = dict(g=None if g is None else flat(g), beta=None if beta is None else flat(beta))
    else:
        state0 = h0
        args = [q, k, v]
        kw = {}
        gates = dict(g=g, beta=beta)
    scale = meta["scale"]
    if meta.get("l2norm"):
        kw["use_qk_l2norm_in_kernel"] = True
    if kind == "gated_deltanet":
        if impl == "flashqla":
            from flash_qla import chunk_gated_delta_rule as fn
        else:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fn
        o, ht = fn(*args, g=gates["g"], beta=gates["beta"], scale=scale, initial_state=state0,
                   output_final_state=True, **kw)
    elif kind == "deltanet":
        # chunk_delta_rule has no grouped key heads (FLA main returns inf/NaN for them instead of raising).
        from fla.ops.delta_rule import chunk_delta_rule as fn
        args = [repeat_heads_f32(args[0], heads), repeat_heads_f32(args[1], heads), args[2]]
        o, ht = fn(*args, beta=gates["beta"], scale=scale, initial_state=state0, output_final_state=True, **kw)
    elif kind == "simple_gla":
        from fla.ops.simple_gla import chunk_simple_gla as fn
        args = [repeat_heads_f32(args[0], heads), repeat_heads_f32(args[1], heads), args[2]]
        o, ht = fn(*args, g=gates["g"], scale=scale, initial_state=state0, output_final_state=True, **kw)
    else:
        raise ValueError(kind)
    o = o.reshape(batch, seq_len, heads, value_dim)
    if seg is not None:
        ht = ht.index_select(0, torch.tensor(last, device=DEV))
    return backward(o, ht, leaves, case)


def import_mamba_ssd():
    """``mamba_chunk_scan_combined`` without the package ``__init__`` (it imports the TileLang Mamba-3 ops)."""
    import importlib.util
    import sys
    import types

    if "mamba_ssm" not in sys.modules:
        spec = importlib.util.find_spec("mamba_ssm")
        package = types.ModuleType("mamba_ssm")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["mamba_ssm"] = package
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    return mamba_chunk_scan_combined


def run_fla_ssd(case: dict, meta: dict) -> dict:
    """FLA ``chunk_simple_gla`` through the SSD mapping (as our side runs it: ``v = dt x`` rounded to bf16)."""
    from fla.ops.simple_gla import chunk_simple_gla

    dtype = getattr(torch, meta["dtype"])
    x = torch.tensor(case["x"], device=DEV, dtype=dtype, requires_grad=True)
    bm = torch.tensor(case["B"], device=DEV, dtype=dtype, requires_grad=True)
    cm = torch.tensor(case["C"], device=DEV, dtype=dtype, requires_grad=True)
    dt = torch.tensor(case["dt"], device=DEV, dtype=torch.float32, requires_grad=True)
    a = torch.tensor(case["A"], device=DEV, dtype=torch.float32, requires_grad=True)
    h0 = torch.tensor(case["h0"], device=DEV, dtype=torch.float32, requires_grad=True) if "h0" in case else None
    batch, seq_len, heads, _ = x.shape
    # Group sums of the repeated B/C in float32, as our layer now does.
    q = repeat_heads(cm.float(), heads).to(dtype)
    k = repeat_heads(bm.float(), heads).to(dtype)
    v = (dt[..., None] * x.float()).to(dtype)
    g = dt * a
    state0 = None if h0 is None else h0.transpose(-1, -2)
    seg = case.get("seg")
    if seg is not None:
        docs = docs_of(seg)
        rows = [(b, r) for b, rr in enumerate(docs) for r in rr]
        cu = torch.tensor([b * seq_len + r[0] for b, r in rows] + [batch * seq_len], device=DEV, dtype=torch.long)
        first = [i for i, (_, r) in enumerate(rows) if r[0] == 0]
        last = [i for i, (_, r) in enumerate(rows) if r[1] == seq_len]
        if state0 is not None:
            s0 = torch.zeros(len(rows), *state0.shape[1:], device=DEV, dtype=torch.float32)
            state0 = s0.index_copy(0, torch.tensor(first, device=DEV), state0)

        def flat(t):
            return t.reshape(1, batch * seq_len, *t.shape[2:])

        o, ht = chunk_simple_gla(flat(q), flat(k), flat(v), g=flat(g), scale=1.0, initial_state=state0,
                                 output_final_state=True, cu_seqlens=cu)
        ht = ht.index_select(0, torch.tensor(last, device=DEV))
    else:
        o, ht = chunk_simple_gla(q, k, v, g=g, scale=1.0, initial_state=state0, output_final_state=True)
    o = o.reshape(x.shape)
    leaves = dict(x=x, dt=dt, A=a, B=bm, C=cm)
    if h0 is not None:
        leaves["h0"] = h0
    return backward(o, ht.transpose(-1, -2), leaves, case)


def run_mamba2(case: dict, meta: dict) -> dict:
    mamba_chunk_scan_combined = import_mamba_ssd()

    dtype = getattr(torch, meta["dtype"])
    x = torch.tensor(case["x"], device=DEV, dtype=dtype, requires_grad=True)
    bm = torch.tensor(case["B"], device=DEV, dtype=dtype, requires_grad=True)
    cm = torch.tensor(case["C"], device=DEV, dtype=dtype, requires_grad=True)
    dt = torch.tensor(case["dt"], device=DEV, dtype=torch.float32, requires_grad=True)
    a = torch.tensor(case["A"], device=DEV, dtype=torch.float32, requires_grad=True)
    h0 = torch.tensor(case["h0"], device=DEV, dtype=torch.float32, requires_grad=True) if "h0" in case else None
    seq_idx = torch.tensor(case["seg"], device=DEV, dtype=torch.int32) if "seg" in case else None
    if seq_idx is not None:  # seq_idx must start at 0 in every row and increase by one per document
        seq_idx = seq_idx - seq_idx[:, :1]
    o, ht = mamba_chunk_scan_combined(x, dt, a, bm, cm, chunk_size=meta.get("mamba_chunk", 256),
                                      initial_states=h0, seq_idx=seq_idx, return_final_states=True)
    leaves = dict(x=x, dt=dt, A=a, B=bm, C=cm)
    if h0 is not None:
        leaves["h0"] = h0
    return backward(o, ht, leaves, case)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--impls", required=True)
    parser.add_argument("--tag", default="", help="suffix for the output impl name (e.g. _full)")
    args = parser.parse_args()
    for name in args.cases.split(","):
        case = load(os.path.join(args.dir, f"{name}.in.npz"))
        meta = meta_of(case)
        for impl in args.impls.split(","):
            if impl in ("mamba2", "fla_ssd") and meta["kind"] != "mamba2":
                continue
            if impl in ("fla", "flashqla") and meta["kind"] == "mamba2":
                continue
            if impl == "flashqla" and (meta["kind"] != "gated_deltanet" or meta["dtype"] != "bfloat16"):
                continue
            if impl == "oracle" and meta.get("no_oracle"):
                continue
            t0 = time.time()
            out_path = os.path.join(args.dir, f"{name}.{impl}{args.tag}.npz")
            try:
                if impl == "oracle":
                    res = run_oracle(case, meta)
                elif impl == "mamba2":
                    res = run_mamba2(case, meta)
                elif impl == "fla_ssd":
                    res = run_fla_ssd(case, meta)
                else:
                    res = run_fla_like(case, meta, impl)
                torch.cuda.synchronize()
                np.savez(out_path, **res)
                print(json.dumps(dict(case=name, impl=impl + args.tag, ok=True, s=round(time.time() - t0, 1))),
                      flush=True)
            except Exception as err:  # noqa: BLE001 (record and continue)
                np.savez(out_path, error=np.array(f"{type(err).__name__}: {err}"[:2000]))
                print(json.dumps(dict(case=name, impl=impl + args.tag, ok=False,
                                      error=f"{type(err).__name__}: {str(err)[:300]}")), flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
