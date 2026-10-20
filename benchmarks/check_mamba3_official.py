# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Differential check of Mamba-3 (SISO and MIMO) against the official state-spaces/mamba code.

Two processes see identical inputs:

    python benchmarks/check_mamba3_official.py torch --dir <dir>     # official (torch venv)
    python benchmarks/check_mamba3_official.py jax --dir <dir> --out <json>

The torch side writes one ``<case>.npz`` per case with the inputs, the official outputs, every
gradient and the final states. It calls the official kernels (``mamba3_siso_combined``, the TileLang
``mamba3_mimo``) and the official ``Mamba3`` module (with ``step`` for decoding). The jax side runs ours
on the same inputs: every ``impl`` of ``jla.layers.ssm.mamba3_siso`` / ``jla.layers.mamba3_mimo.mamba3_mimo``
and the ``Mamba3`` / ``Mamba3MIMO`` layers with the official weights. It also runs a float32 oracle (our
token-by-token reference with float32 inputs) and reports ``max|a - b| / max|b|`` per tensor for ours
vs official, ours vs float32 and official vs float32.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------- cases

# Op cases: B, T, H, G (B/C groups), N (d_state), P (head_dim), S (angles), and options.
SISO_OPS = {
    "siso_dense": dict(B=2, T=512, H=4, G=1, N=128, P=64, S=32),
    "siso_odd_gqa_64x128": dict(B=2, T=333, H=8, G=2, N=64, P=128, S=16),
    "siso_64x64_rope_full": dict(B=2, T=200, H=4, G=1, N=64, P=64, S=32),
    "siso_init_final": dict(B=2, T=200, H=4, G=1, N=128, P=64, S=32, init=True, final=True),
    "siso_varlen": dict(B=1, T=700, H=4, G=1, N=128, P=64, S=32, cu=[0, 64, 65, 300, 477, 700], final=True),
    "siso_extreme": dict(B=2, T=256, H=4, G=1, N=128, P=64, S=32, extreme=True),
    "siso_no_dz": dict(B=2, T=130, H=4, G=1, N=128, P=64, S=32, dz=False),
}
MIMO_OPS = {
    "mimo_r4": dict(B=2, T=512, H=4, G=1, R=4, N=128, P=64, S=32, final=True),
    # The official dense MIMO path needs T % (64 / R) == 0; odd lengths go through its varlen kernels.
    "mimo_r2_odd": dict(B=1, T=333, H=4, G=1, R=2, N=128, P=64, S=32, cu=[0, 333], final=True),
    "mimo_r1": dict(B=2, T=256, H=4, G=1, R=1, N=128, P=64, S=32),
    "mimo_r4_gqa_64x64": dict(B=1, T=256, H=8, G=2, R=4, N=64, P=64, S=16),
    "mimo_r4_varlen": dict(B=1, T=600, H=4, G=1, R=4, N=128, P=64, S=32, cu=[0, 100, 101, 357, 600], final=True),
    "mimo_r2_rank_out": dict(B=2, T=256, H=4, G=1, R=2, N=128, P=64, S=32, out=False, z=False),
    "mimo_r4_extreme": dict(B=2, T=256, H=4, G=1, R=4, N=128, P=64, S=32, extreme=True),
    "mimo_r2_rope_full_64x64": dict(B=2, T=128, H=4, G=1, R=2, N=64, P=64, S=32),  # rotary_dim_divisor 2
}
# Layer cases: the official Mamba3 module vs ours with the same weights; ``decode`` tokens after a
# prefill of ``T - decode`` tokens go through the official ``step`` and our ``step``.
LAYERS = {
    "layer_siso": dict(D=256, N=128, P=64, G=1, R=1, rope=0.5, norm=False, chunk=64, B=2, T=160, decode=4),
    "layer_siso_norm_rope_full": dict(D=256, N=64, P=64, G=1, R=1, rope=1.0, norm=True, chunk=64, B=2, T=130,
                                      decode=3),
    "layer_mimo_r4": dict(D=256, N=128, P=64, G=1, R=4, rope=0.5, norm=False, chunk=16, B=2, T=160, decode=4),
    # The official dense MIMO backward needs T % (64 / R) == 0.
    "layer_mimo_r2_norm": dict(D=256, N=128, P=64, G=1, R=2, rope=0.5, norm=True, chunk=32, B=2, T=160, decode=3),
}


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bfloat16 (ties to even), kept as float32."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bits = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def _segments(cu: list[int]) -> np.ndarray:
    seg = np.zeros((1, cu[-1]), np.int32)
    for i, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        seg[0, a:b] = i
    return seg


def op_inputs(c: dict, mimo: bool, seed: int) -> dict:
    """Raw inputs in the official layouts (float32 numpy, bf16-representable where the kernels read bf16)."""
    rng = np.random.default_rng(seed)
    B, T, H, G, N, P, S = (c[n] for n in "BTHGNPS")
    R = c.get("R", 1)
    ext = c.get("extreme", False)
    normal = lambda *s: rng.standard_normal(s, dtype=np.float32)  # noqa: E731

    def rms(*s):
        x = normal(*s)
        return _bf16(x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6))

    qk_shape = (B, T, R, G, N) if mimo else (B, T, G, N)
    lo, hi = (1e-2, 1.0) if ext else (1e-3, 1e-1)
    dt = np.exp(rng.uniform(math.log(lo), math.log(hi), (B, H, T))).astype(np.float32)
    a = -rng.uniform(1.0, 64.0 if ext else 16.0, (B, H, T)).astype(np.float32)
    x = dict(
        q=rms(*qk_shape), k=rms(*qk_shape), v=_bf16(normal(B, T, H, P)), adt=a * dt, dt=dt,
        trap=_bf16(normal(B, H, T) * (8.0 if ext else 1.0)),
        q_bias=normal(*((H, R, N) if mimo else (H, N))), k_bias=normal(*((H, R, N) if mimo else (H, N))),
        angles=_bf16(normal(B, T, H, S) * (4.0 if ext else 1.0)),  # the official SISO wrapper rounds to bf16
    )
    if c.get("dz", True):
        x["d"] = 1.0 + 0.5 * normal(H)
    if c.get("z", c.get("dz", True)):
        x["z"] = _bf16(normal(B, T, H, P))
    if mimo:
        x["mimo_x"] = (1.0 + 0.3 * normal(H, R, P)) / R
        x["mimo_z"] = 1.0 + 0.3 * normal(H, R, P)
        if c.get("out", True):
            x["mimo_o"] = (1.0 + 0.3 * normal(H, R, P)) / R
    if c.get("init"):
        x["init_angle"] = rng.uniform(0.0, 2 * math.pi, (B, H, S)).astype(np.float32)
        x["init_ssm"] = 0.1 * normal(B, H, P, N)
        x["init_k"] = _bf16(normal(B, H, N))
        x["init_v"] = _bf16(normal(B, H, P))
    do_shape = (B, T, H, P) if (not mimo or c.get("out", True)) else (B, T, R, H, P)
    x["do"] = _bf16(normal(*do_shape))
    return x


# ------------------------------------------------------------------------------------- torch


def run_torch(args) -> None:
    import torch

    from mamba_ssm.modules.mamba3 import Mamba3
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from mamba_ssm.utils.generation import InferenceParams

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    dev = torch.device("cuda")
    # The official module passes Trap in bf16 (an in_proj output). The SISO wrapper casts it to bf16, but
    # the TileLang MIMO kernels declare TRAP in the compute dtype and read a float32 tensor as garbage.
    bf16_names = ("q", "k", "v", "z", "do", "init_k", "init_v", "trap")
    args.dir.mkdir(parents=True, exist_ok=True)

    def tensors(raw):
        t = {}
        for n, a in raw.items():
            t[n] = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
            if n in bf16_names:
                t[n] = t[n].to(torch.bfloat16)
        return t

    def grads_of(out, leaves, do):
        names = list(leaves)
        gs = torch.autograd.grad(out, [leaves[n] for n in names], grad_outputs=do, allow_unused=True)
        return {f"d{n}": g.float().cpu().numpy() for n, g in zip(names, gs) if g is not None}

    for seed, (name, c) in enumerate(list(SISO_OPS.items()) + list(MIMO_OPS.items())):
        if args.cases and name not in args.cases:
            continue
        mimo = name.startswith("mimo")
        raw = op_inputs(c, mimo, seed)
        x = tensors(raw)
        leaves = {n: x[n].clone().requires_grad_(True) for n in x if n != "do"}
        cu = torch.tensor(c["cu"], dtype=torch.int32, device=dev) if "cu" in c else None
        final = c.get("final", False)
        t0 = time.time()
        if mimo:
            R = c["R"]
            out = mamba3_mimo(
                leaves["q"], leaves["k"], leaves["v"], leaves["adt"], leaves["dt"], leaves["trap"],
                leaves["q_bias"], leaves["k_bias"], leaves["mimo_x"], leaves["mimo_z"], leaves.get("mimo_o"),
                leaves["angles"], leaves.get("d"), leaves.get("z"), chunk_size=64 // R,
                rotary_dim_divisor=c["N"] // c["S"], dtype=torch.bfloat16, return_state=final,
                cu_seqlens=cu,
            )
        else:
            init = None
            if c.get("init"):
                init = tuple(leaves[n] for n in ("init_angle", "init_ssm", "init_k", "init_v"))
            out = mamba3_siso_combined(
                leaves["q"], leaves["k"], leaves["v"], leaves["adt"], leaves["dt"], leaves["trap"],
                leaves["q_bias"], leaves["k_bias"], leaves["angles"], D=leaves.get("d"), Z=leaves.get("z"),
                Input_States=init, return_final_states=final, cu_seqlens=cu,
            )
        dump = {f"in_{n}": a for n, a in raw.items()}
        if final:
            out, *states = out
            for n, s in zip(("angle", "ssm", "k", "v"), states):
                dump[f"final_{n}"] = s.detach().float().cpu().numpy()
        dump["out"] = out.detach().float().cpu().numpy()
        dump.update(grads_of(out, leaves, x["do"]))
        np.savez(args.dir / f"{name}.npz", **dump)
        print(f"{name}: official out {tuple(out.shape)} in {time.time() - t0:.1f} s; grads "
              f"{sorted(k for k in dump if k.startswith('d'))}", flush=True)

    for seed, (name, c) in enumerate(LAYERS.items(), start=100):
        if args.cases and name not in args.cases:
            continue
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        R = c["R"]
        m = Mamba3(c["D"], d_state=c["N"], expand=2, headdim=c["P"], ngroups=c["G"], rope_fraction=c["rope"],
                   is_outproj_norm=c["norm"], is_mimo=R > 1, mimo_rank=R, chunk_size=c["chunk"], layer_idx=0,
                   device=dev, dtype=torch.bfloat16)
        dump = {}
        with torch.no_grad():  # random values everywhere so that a wrong mapping cannot hide
            for pname, p in m.named_parameters():
                if pname.endswith("proj.weight"):
                    val = rng.standard_normal(p.shape, dtype=np.float32) / math.sqrt(p.shape[1])
                elif pname in ("mimo_x", "mimo_o"):
                    val = (1.0 + 0.3 * rng.standard_normal(p.shape, dtype=np.float32)) / R
                elif pname == "dt_bias":
                    val = p.detach().float().cpu().numpy()
                else:
                    val = 1.0 + 0.3 * rng.standard_normal(p.shape, dtype=np.float32)
                val = _bf16(val) if p.dtype == torch.bfloat16 else val.astype(np.float32)
                p.copy_(torch.from_numpy(val).to(dev, p.dtype))
                dump[f"p_{pname}"] = val
        B, T, decode = c["B"], c["T"], c["decode"]
        u = _bf16(rng.standard_normal((B, T, c["D"]), dtype=np.float32))
        do = _bf16(rng.standard_normal((B, T, c["D"]), dtype=np.float32))
        dump["in_u"], dump["in_do"] = u, do
        ut = torch.from_numpy(u).to(dev, torch.bfloat16).requires_grad_(True)
        out = m(ut)
        params = dict(m.named_parameters())
        gs = torch.autograd.grad(out, [ut] + list(params.values()), grad_outputs=torch.from_numpy(do).to(dev, torch.bfloat16))
        dump["out"] = out.detach().float().cpu().numpy()
        dump["du"] = gs[0].float().cpu().numpy()
        for pname, g in zip(params, gs[1:]):
            dump[f"dp_{pname}"] = g.float().cpu().numpy()
        if decode:
            states = m.allocate_inference_cache(B, T)
            ip = InferenceParams(max_seqlen=T, max_batch_size=B)
            ip.key_value_memory_dict[0] = states
            with torch.no_grad():
                pre = m(ut[:, : T - decode].detach(), inference_params=ip)
            dump["prefill"] = pre.float().cpu().numpy()
            for n, s in zip(("angle", "ssm", "k", "v"), states):  # the cache after the prefill
                dump[f"state_{n}"] = s.float().cpu().numpy()
            try:  # Mamba3.step needs the CuteDSL step kernel (nvidia-cutlass-dsl, quack-kernels)
                with torch.no_grad():
                    steps = [m.step(ut[:, t].detach(), *states)[0] for t in range(T - decode, T)]
                dump["decode"] = torch.stack(steps, 1).float().cpu().numpy()
            except Exception as err:  # noqa: BLE001 - report and keep the forward/backward dump
                print(f"{name}: official Mamba3.step failed ({type(err).__name__}: {str(err)[:160]}); "
                      "decode is compared with the official full forward only", flush=True)
        np.savez(args.dir / f"{name}.npz", **dump)
        print(f"{name}: official module out {tuple(out.shape)}; {len(params)} parameters", flush=True)


# --------------------------------------------------------------------------------------- jax


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape {a.shape} vs {b.shape}")
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def run_jax(args) -> dict:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jla.layers import ssm as ssm_layer
    from jla.layers.mamba3_mimo import Mamba3MIMO, mamba3_mimo
    from jla.layers.ssm import Mamba3, mamba3_siso
    from jla.models.config import SSMConfig
    from jla.ops.reference import Mamba3State

    f32, bf16 = jnp.float32, jnp.bfloat16
    report: dict = {}
    bf16_names = ("q", "k", "v", "z", "do", "init_k", "init_v")

    def ours_op(name, c, x, impl, dtype):
        """(y, grads dict, final state dict) of our op on the official inputs (our layouts)."""
        mimo = name.startswith("mimo")
        cast = lambda n: jnp.asarray(x[n], dtype if n in bf16_names else f32)  # noqa: E731
        tr = lambda a: jnp.swapaxes(a, 1, 2)  # [B, H, T] -> [B, T, H]  # noqa: E731
        seg = jnp.asarray(_segments(c["cu"])) if "cu" in c else None
        final = c.get("final", False)
        names = [n for n in x if n != "do" and not n.startswith("init")] + (
            ["init_angle", "init_ssm", "init_k", "init_v"] if c.get("init") else [])
        leaves = {n: cast(n) for n in names}
        for n in ("adt", "dt", "trap"):
            leaves[n] = tr(leaves[n])
        if "init_ssm" in leaves:
            leaves["init_ssm"] = jnp.swapaxes(leaves["init_ssm"], 2, 3)  # [B, H, P, N] -> [B, H, N, P]
        kernel_impl = "mosaic_gpu" if impl.startswith("mosaic_gpu") else impl

        def fn(lv):
            common = dict(d=lv.get("d"), z=lv.get("z"), segment_ids=seg, output_final_state=final, impl=kernel_impl)
            if mimo:
                R = c["R"]
                return mamba3_mimo(lv["q"], lv["k"], lv["v"], lv["adt"], lv["dt"], lv["trap"], lv["q_bias"],
                                   lv["k_bias"], lv["angles"], lv["mimo_x"],
                                   mimo_z=lv["mimo_z"] if "z" in lv else None, mimo_o=lv.get("mimo_o"),
                                   chunk_size=64 // R if kernel_impl == "mosaic_gpu" else 16, **common)
            init = None
            if "init_ssm" in lv:
                init = (lv["init_angle"], Mamba3State(lv["init_ssm"], lv["init_k"].astype(f32),
                                                      lv["init_v"].astype(f32)))
            saved = ssm_layer._fused_core
            if impl == "mosaic_gpu+xla":
                ssm_layer._fused_core = lambda: None
            try:
                return mamba3_siso(lv["q"], lv["k"], lv["v"], lv["adt"], lv["dt"], lv["trap"], lv["q_bias"],
                                   lv["k_bias"], lv["angles"], initial_state=init, **common)
            finally:
                ssm_layer._fused_core = saved

        if mimo and "z" not in leaves:
            leaves.pop("mimo_z", None)

        def run(lv, do):
            (y, state), vjp = jax.vjp(fn, lv)
            zero_state = jax.tree_util.tree_map(jnp.zeros_like, state)
            (g,) = vjp((do.astype(y.dtype), zero_state))
            return y, g, state

        do = cast("do")
        if mimo and not c.get("out", True):
            do = jnp.moveaxis(do, 2, 3)  # official [B, T, R, H, P] -> ours [B, T, H, R, P]
        y, g, state = jax.jit(run)(leaves, do)
        grads = {}
        for n, v in g.items():
            v = np.asarray(v, np.float32)
            if n in ("adt", "dt", "trap"):
                v = np.swapaxes(v, 1, 2)
            if n == "init_ssm":
                v = np.swapaxes(v, 2, 3)
            grads[f"d{n}"] = v
        out = np.asarray(y, np.float32)
        if mimo and not c.get("out", True):
            out = np.moveaxis(out, 3, 2)  # ours [B, T, H, R, P] -> official [B, T, R, H, P]
        states = {}
        if final:
            angle, st = state
            states["final_angle"] = np.mod(np.asarray(angle, np.float32), 2 * np.pi)
            states["final_ssm"] = np.swapaxes(np.asarray(st.ssm, np.float32), -1, -2)  # -> [B, H, P, N]
            if mimo:
                states["final_k"] = np.swapaxes(np.asarray(st.key, np.float32), 1, 2)  # -> [B, R, H, N]
                states["final_v"] = np.asarray(st.value, np.float32)  # [B, H, R, P] = x * mimo_x
            else:
                states["final_k"] = np.asarray(st.key, np.float32)
                states["final_v"] = np.asarray(st.value, np.float32)
        return out, grads, states

    def official_states(want, c, x):
        """The official final states in our comparison layout (last sequence of a packed row)."""
        st = {k: want[k] for k in ("final_angle", "final_ssm", "final_k", "final_v") if k in want}
        if not st:
            return st
        if "cu" in c:
            st = {k: v[-1:] for k, v in st.items()}
        st["final_angle"] = np.mod(st["final_angle"], 2 * np.pi)
        if "mimo_x" in x:  # official keeps the raw last x; ours the up-projected x * mimo_x per rank
            st["final_v"] = st["final_v"][:, :, None, :] * x["mimo_x"][None]
        return st

    for name, c in list(SISO_OPS.items()) + list(MIMO_OPS.items()):
        if args.cases and name not in args.cases:
            continue
        path = args.dir / f"{name}.npz"
        if not path.exists():
            continue
        want = dict(np.load(path))
        x = {k[3:]: v for k, v in want.items() if k.startswith("in_")}
        mimo = name.startswith("mimo")
        impls = ["reference", "chunked", "mosaic_gpu"] + ([] if mimo else ["mosaic_gpu+xla"])
        oracle = ours_op(name, c, x, "reference", f32)
        ref_states = official_states(want, c, x)
        row = report[name] = dict(case=c, official_vs_f32={}, impls={})
        row["official_vs_f32"]["out"] = rel(want["out"], oracle[0])
        for k, v in oracle[1].items():
            if k in want:
                row["official_vs_f32"][k] = rel(want[k], v)
        for k, v in ref_states.items():
            row["official_vs_f32"][k] = (angle_rel if k == "final_angle" else rel)(v, oracle[2][k])
        for impl in impls:
            try:
                got = ours_op(name, c, x, impl, bf16)
            except Exception as err:  # noqa: BLE001 - unsupported combination: record it
                row["impls"][impl] = dict(error=f"{type(err).__name__}: {str(err)[:200]}")
                print(f"{name} {impl}: {row['impls'][impl]['error']}", flush=True)
                continue
            vs_off, vs_f32 = {"out": rel(got[0], want["out"])}, {"out": rel(got[0], oracle[0])}
            for k, v in got[1].items():
                vs_f32[k] = rel(v, oracle[1][k])
                if k in want:
                    vs_off[k] = rel(v, want[k])
                else:
                    vs_off[k] = "official has no gradient"
            for k, v in ref_states.items():
                f = angle_rel if k == "final_angle" else rel
                vs_off[k], vs_f32[k] = f(got[2][k], v), f(got[2][k], oracle[2][k])
            row["impls"][impl] = dict(vs_official=vs_off, vs_f32=vs_f32)
            worst = max((v, k) for k, v in vs_off.items() if isinstance(v, float))
            print(f"{name} {impl}: worst vs official {worst[1]} {worst[0]:.2e}; vs f32 out {vs_f32['out']:.2e}",
                  flush=True)
        worst_off = max((v, k) for k, v in row["official_vs_f32"].items())
        print(f"{name} official vs f32: worst {worst_off[1]} {worst_off[0]:.2e}", flush=True)

    for name, c in LAYERS.items():
        if args.cases and name not in args.cases:
            continue
        path = args.dir / f"{name}.npz"
        if not path.exists():
            continue
        want = dict(np.load(path))
        row = report[name] = dict(case=c, impls={})
        for impl, compute in (("reference-f32", "float32"), ("reference", "bfloat16"), ("chunked", "bfloat16"),
                              ("mosaic_gpu", "bfloat16"), ("auto", "bfloat16")):
            try:
                row["impls"][impl] = layer_check(jax, jnp, eqx, SSMConfig, Mamba3, Mamba3MIMO, c, want,
                                                 impl.split("-")[0], compute)
            except Exception as err:  # noqa: BLE001
                row["impls"][impl] = dict(error=f"{type(err).__name__}: {str(err)[:300]}")
            r = row["impls"][impl]
            if "error" in r:
                print(f"{name} {impl}: {r['error']}", flush=True)
            else:
                worst = max((v, k) for k, v in r["vs_official"].items())
                print(f"{name} {impl}: worst vs official {worst[1]} {worst[0]:.2e}; out {r['vs_official']['out']:.2e}",
                      flush=True)
    return report


def layer_check(jax, jnp, eqx, SSMConfig, Mamba3, Mamba3MIMO, c, want, impl, compute) -> dict:
    """Our layer with the official module's weights: output, input and parameter gradients, decode."""
    R, N, P, G, D = c["R"], c["N"], c["P"], c["G"], c["D"]
    cfg = SSMConfig(d_state=N, expand=2, head_dim=P, n_groups=G, rope_fraction=c["rope"], outproj_norm=c["norm"],
                    chunk_size=c["chunk"], implementation=impl, mimo_rank=R)
    cls = Mamba3MIMO if R > 1 else Mamba3
    layer = cls(D, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5, compute_dtype=compute)
    inner, H = 2 * D, 2 * D // P
    M = N * G * R
    W = want["p_in_proj.weight"]  # [d_in_proj, D] rows [z, x, B, C, dd_dt, dd_A, trap, angles]
    rows = np.cumsum([0, inner, inner, M, M, H, H, H])

    def ours_from(get, prefix):
        """Official tensors (weights or their gradients) in our parameter layout."""
        w = get(f"{prefix}in_proj.weight")
        blk = lambda i: w[rows[i]:rows[i + 1]].T  # noqa: E731
        out = {
            "zx_proj.weight": np.stack([blk(0), blk(1)], 1),
            "bc_proj.weight": np.stack([blk(2), blk(3)], 1),
            "scalar_proj.weight": np.stack([blk(4), blk(5), blk(6)], 1),
            "angle_proj.weight": w[rows[7]:].T,
            "dt_bias": get(f"{prefix}dt_bias"),
            "B_bias": get(f"{prefix}B_bias") if R > 1 else get(f"{prefix}B_bias")[:, 0],
            "C_bias": get(f"{prefix}C_bias") if R > 1 else get(f"{prefix}C_bias")[:, 0],
            "B_norm.weight": get(f"{prefix}B_norm.weight"),
            "C_norm.weight": get(f"{prefix}C_norm.weight"),
            "D": get(f"{prefix}D"),
            "out_proj.weight": get(f"{prefix}out_proj.weight").T,
        }
        if R > 1:
            for n in ("mimo_x", "mimo_z", "mimo_o"):
                out[n] = get(f"{prefix}{n}")
        if c["norm"]:
            out["norm.weight"] = get(f"{prefix}norm.weight")
        return out

    params = ours_from(lambda k: want[k], "p_")

    def where(m, name):
        obj = m
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    names = sorted(params)
    layer = eqx.tree_at(lambda m: [where(m, n) for n in names], layer,
                        [jnp.asarray(params[n], jnp.float32) for n in names])
    u = jnp.asarray(want["in_u"], jnp.bfloat16)
    do = jnp.asarray(want["in_do"], jnp.bfloat16)
    dyn, static = eqx.partition(layer, eqx.is_inexact_array)

    def loss(dyn, u):
        out = eqx.combine(dyn, static)(u)
        return out

    @jax.jit
    def fwd_bwd(dyn, u, do):
        out, vjp = jax.vjp(loss, dyn, u)
        gd, gu = vjp(do.astype(out.dtype))
        return out, gd, gu

    out, gd, gu = fwd_bwd(dyn, u, do)
    res = {"out": rel(out, want["out"]), "du": rel(gu, want["du"])}
    grads = ours_from(lambda k: want[k], "dp_")
    for n in names:
        res[f"d{n}"] = rel(where(gd, n), grads[n])
    if c["decode"]:
        T, dec = c["T"], c["decode"]
        cache = layer.init_cache(c["B"], jnp.float32)
        step = jax.jit(lambda l, x, ca: l.step(x, ca))
        pre, cache = step(layer, u[:, : T - dec], cache)
        res["prefill"] = rel(pre, want["prefill"])
        st = cache.state
        key, value = np.asarray(st.key, np.float32), np.asarray(st.value, np.float32)
        res["prefill_state_angle"] = angle_rel(np.asarray(cache.angle), want["state_angle"])
        res["prefill_state_ssm"] = rel(np.swapaxes(np.asarray(st.ssm, np.float32), -1, -2), want["state_ssm"])
        if R > 1:  # official k [B, R, H, N] and the raw last x [B, H, P]; ours per rank x * mimo_x
            res["prefill_state_k"] = rel(np.swapaxes(key, 1, 2), want["state_k"])
            res["prefill_state_v"] = rel(value, want["state_v"][:, :, None, :] * want["p_mimo_x"][None])
        else:
            res["prefill_state_k"] = rel(key, want["state_k"][:, 0])
            res["prefill_state_v"] = rel(value, want["state_v"])
        outs = []
        for t in range(T - dec, T):
            o, cache = step(layer, u[:, t: t + 1], cache)
            outs.append(np.asarray(o, np.float32)[:, 0])
        if "decode" in want:
            res["decode"] = rel(np.stack(outs, 1), want["decode"])
        res["decode_vs_full_official"] = rel(np.stack(outs, 1), want["out"][:, T - dec:])
    return dict(vs_official=res)


def angle_rel(a, b) -> float:
    """Largest angle difference modulo 2 pi, as a fraction of pi."""
    d = np.abs(np.mod(np.asarray(a, np.float64) - np.asarray(b, np.float64) + np.pi, 2 * np.pi) - np.pi)
    return float(d.max() / np.pi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("framework", choices=["jax", "torch"])
    parser.add_argument("--dir", type=Path, required=True, help="directory of the per-case npz files")
    parser.add_argument("--out", type=Path, help="jax: write the report as JSON")
    parser.add_argument("--cases", nargs="*", default=[], help="only these cases")
    parser.add_argument("--memory-fraction", type=float, default=0.2, help="torch per-process memory cap")
    args = parser.parse_args()
    if args.framework == "torch":
        run_torch(args)
        return
    report = run_jax(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), cases=report), indent=1))


if __name__ == "__main__":
    main()
