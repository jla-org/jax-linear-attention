# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2025 The JAX Authors.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py (the FP8 GEMM variant)
# and, for the PDL launch check (_LAUNCH_RE), tests/pallas/mosaic_gpu_test.py (test_griddepcontrol_multi_kernel)
"""Probe which Hopper techniques JAX 0.11.2's Mosaic GPU runs, and what they buy.

1. GEMMs at the 1.3B presets' projection shapes (M = 16384 tokens):
   * cuBLAS through XLA in bf16, and in FP8 e4m3 with a bf16 output, both as ``x @ w`` with
     ``w [K, N]`` (XLA inserts the transpose cuBLASLt FP8 needs) and with ``w`` stored ``[N, K]``;
   * JAX's shipped persistent, warp-specialized Mosaic GPU matmul
     (``jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu``) in bf16, without clusters and with
     2-CTA clusters whose shared operand is TMA-multicast (``cluster_dimension`` M or N);
   * an FP8 e4m3 variant of that kernel (both operands K-major, as Hopper's 8-bit WGMMA requires).
   Every result is checked against a float32 product of the same (quantized) inputs.
2. Programmatic dependent launch: a jitted chain of small Mosaic GPU kernels, with and without
   ``griddepcontrol.wait`` / ``griddepcontrol.launch_dependents``; wall time per kernel. The
   griddepcontrol instructions are no-ops unless the kernel is launched with the programmatic
   stream serialization attribute, so the probe also reads Mosaic GPU's host-side LLVM dump
   (as JAX's ``test_griddepcontrol_multi_kernel`` does) and records the ``uses_pdl`` argument
   each kernel launch passes to ``mosaic_gpu_launch_kernel``.

    python benchmarks/bench_hopper_features.py --json benchmarks/results/hopper_features.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import statistics
import sys
import re
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import jax.experimental.pallas as pl  # noqa: E402
import jax.experimental.pallas.mosaic_gpu as plgpu  # noqa: E402
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm  # noqa: E402

# (M, K, N): tokens x in-features x out-features of one GEMM at the 1.3B presets (d_model 2048).
SHAPES = {
    "mlp_gate_up": (16384, 2048, 11264),   # SwiGLU gate+up, d_ff 5632 x 2
    "mlp_down": (16384, 5632, 2048),
    "attn_qkv_gqa": (16384, 2048, 3072),   # 32 q heads + 2 x 8 kv heads of 64
    "attn_o": (16384, 2048, 2048),
    "lm_head_tile": (16384, 2048, 16384),  # one vocabulary tile of the 128,256-entry head
    "quick": (2048, 2048, 4096),           # --quick: does it compile and agree (not a timing)
}
ITERS = harness.ITERATIONS


def _err(out, ref) -> float:
    out = np.asarray(out, np.float32)
    ref = np.asarray(ref, np.float32)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def _time(fn, *args) -> dict:
    try:
        return harness.median_kernel_ms(fn, *args, iterations=ITERS)
    except Exception as e:  # report and continue: the probe asks what runs
        return dict(error=harness.error_summary(f"{type(e).__name__}: {e}"))


def _tflops(m, k, n, res):
    if "median_ms" in res:
        res["tflops"] = round(2.0 * m * k * n / (res["median_ms"] * 1e-3) / 1e12, 1)
    return res


# --------------------------------------------------------------------------------------------
# FP8 variant of JAX's Hopper matmul: B is stored [N, K] (K-major) and transposed as a view.
# --------------------------------------------------------------------------------------------
def _fp8_kernel(a_gmem, bt_gmem, out_gmem, config: hm.TuningConfig):
    m, k = a_gmem.shape
    n, k2 = bt_gmem.shape
    assert k == k2
    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(a_gmem.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(a_gmem.dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))
    M, N = hm.MatmulDimension.M, hm.MatmulDimension.N
    cta_tile_m = tile_m * (1 + (config.wg_dimension == M))
    cta_tile_n = tile_n * (1 + (config.wg_dimension == N))
    cluster_tile_m = cta_tile_m * (1 + (config.cluster_dimension == M))
    cluster_tile_n = cta_tile_n * (1 + (config.cluster_dimension == N))
    if m % cluster_tile_m or n % cluster_tile_n or k % tile_k:
        raise ValueError(f"shape {(m, k, n)} does not tile with {config}")
    m_iters, n_iters, k_iters = m // cluster_tile_m, n // cluster_tile_n, k // tile_k
    out_dtype = out_gmem.dtype
    epi_tile_m, epi_tile_n = config.epi_tile_m, config.epi_tile_n
    num_out_slots = min(2, (tile_m * tile_n) // (epi_tile_m * epi_tile_n))
    out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(out_dtype).itemsize * 8)
    out_swizzle_elems = out_swizzle // jnp.dtype(out_dtype).itemsize
    out_transforms = (plgpu.TilingTransform((8, out_swizzle_elems)), plgpu.SwizzleTransform(out_swizzle))

    def get_pipeline(pipeline_body, compute_context):
        return plgpu.emit_pipeline_warp_specialized(
            pipeline_body,
            grid=(k_iters,),
            memory_registers=40,
            in_specs=[
                plgpu.BlockSpec((cta_tile_m, tile_k), lambda kk: (0, kk), transforms=transforms,
                                memory_space=plgpu.SMEM,
                                collective_axes=("cluster",) if config.cluster_dimension == N else ()),
                plgpu.BlockSpec((cta_tile_n, tile_k), lambda kk: (0, kk), transforms=transforms,
                                memory_space=plgpu.SMEM,
                                collective_axes=("cluster",) if config.cluster_dimension == M else ()),
            ],
            wg_axis="wg",
            num_compute_wgs=2,
            max_concurrent_steps=config.max_concurrent_steps,
            compute_context=compute_context,
        )

    ignore = lambda *_, **__: None  # noqa: E731

    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(a_gmem, bt_gmem),
        out_smem=plgpu.SMEM((2, num_out_slots, epi_tile_m, epi_tile_n), out_dtype, transforms=out_transforms),
        collective_axes="wg",
    )
    def _scope(pipeline_allocs, out_smem):
        wg_idx = lax.axis_index("wg")
        cta_idx = lax.axis_index("cluster")

        @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            (lin_idx,) = loop_info.index
            m_idx, n_idx = plgpu.planar_snake(lin_idx, (m_iters, n_iters), config.grid_minor_dim,
                                              config.grid_tile_width)
            if config.cluster_dimension == M:
                m_idx = m_idx * 2 + cta_idx
            elif config.cluster_dimension == N:
                n_idx = n_idx * 2 + cta_idx
            cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
            cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
            if config.wg_dimension == M:
                wg_m_slice, wg_n_slice = pl.ds(wg_idx * tile_m, tile_m), slice(None)
            else:
                wg_m_slice, wg_n_slice = slice(None), pl.ds(wg_idx * tile_n, tile_n)

            def compute_context(eval_pipeline):
                @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32))
                def _acc_scope(acc_ref):
                    eval_pipeline(acc_ref)
                    acc = acc_ref[...].astype(out_dtype)
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    for epi_mi in range(tile_m // epi_tile_m):
                        for epi_ni in range(tile_n // epi_tile_n):
                            epi_m = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
                            epi_n = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            slot = (epi_mi * (tile_n // epi_tile_n) + epi_ni) % 2
                            plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                            out_smem[wg_idx, slot] = acc[epi_m, epi_n]
                            plgpu.commit_smem()
                            plgpu.copy_smem_to_gmem(
                                out_smem.at[wg_idx, slot],
                                out_gmem.at[cta_m_slice, cta_n_slice].at[wg_m_slice, wg_n_slice].at[epi_m, epi_n],
                            )

            def mma_body(idxs, a_smem, b_smem, acc_ref):
                del idxs
                plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem.at[wg_n_slice].transpose((1, 0)))
                plgpu.wgmma_wait(0)
                return acc_ref

            get_pipeline(mma_body, compute_context)(
                a_gmem.at[cta_m_slice, :], bt_gmem.at[cta_n_slice, :], allocations=pipeline_allocs,
            )

    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def fp8_matmul(a, bt, config: hm.TuningConfig, out_dtype=jnp.bfloat16):
    m, _ = a.shape
    n, _ = bt.shape
    num_sms = hm.get_num_sms()
    cluster_size = 1 + (config.cluster_dimension is not None)
    f = plgpu.kernel(
        functools.partial(_fp8_kernel, config=config),
        out_type=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid=(num_sms // cluster_size,),
        grid_names=("cluster_grid",),
        cluster=(cluster_size,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return f(a, bt)


def _configs(tile_k: int):
    out = []
    for wg in (hm.MatmulDimension.N, hm.MatmulDimension.M):
        for cluster in (None, hm.MatmulDimension.M, hm.MatmulDimension.N):
            out.append(hm.TuningConfig(
                tile_m=128, tile_n=128, tile_k=tile_k, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=8, wg_dimension=wg, cluster_dimension=cluster,
            ))
    return out


def _cfg_name(c: hm.TuningConfig) -> str:
    return f"wg={c.wg_dimension} cluster={c.cluster_dimension} tile=({c.tile_m},{c.tile_n},{c.tile_k})"


def _gemm_kinds(hlo: str) -> list[str]:
    """Custom-call targets and Triton GEMM fusions in a compiled module (which library ran the dot)."""
    kinds = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    if '"kind":"__triton_gemm"' in hlo or "__triton_gemm" in hlo:
        kinds.add("__triton_gemm (XLA Triton GEMM fusion)")
    return sorted(kinds)


def _run_mosaic(make, cfg, args, ref, mkn) -> dict:
    """Time one Mosaic GPU config; with 4 pipeline stages over the SMEM budget, retry with 3."""
    res = {}
    for steps in (cfg.max_concurrent_steps, cfg.max_concurrent_steps - 1):
        c = dataclasses.replace(cfg, max_concurrent_steps=steps)
        fn = jax.jit(make(c))
        res = _time(fn, *args)
        res["max_concurrent_steps"] = steps
        if "median_ms" in res:
            res["rel_err"] = _err(fn(*args), ref)
            return _tflops(*mkn, res)
        if "shared memory" not in res.get("error", ""):
            return res
    return res


def gemm_probe(names) -> dict:
    out = {}
    for name in names:
        m, k, n = SHAPES[name]
        ka, kb = jax.random.split(jax.random.key(0))
        a = jax.random.uniform(ka, (m, k), jnp.float32, -1, 1).astype(jnp.bfloat16)
        b = jax.random.uniform(kb, (k, n), jnp.float32, -1, 1).astype(jnp.bfloat16)
        ref = jnp.dot(a.astype(jnp.float32), b.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        rec = dict(shape=dict(m=m, k=k, n=n))
        cublas = jax.jit(lambda x, y: jnp.dot(x, y))
        rec["cublas_bf16"] = _tflops(m, k, n, _time(cublas, a, b))
        rec["cublas_bf16"]["rel_err"] = _err(cublas(a, b), ref)
        rec["cublas_bf16"]["gemm_kinds"] = _gemm_kinds(cublas.lower(a, b).compile().as_text())
        mgpu = {}
        for cfg in _configs(64):
            mgpu[_cfg_name(cfg)] = _run_mosaic(lambda c: functools.partial(hm.matmul, c=None, config=c),
                                               cfg, (a, b), ref, (m, k, n))
        rec["mosaic_bf16"] = mgpu
        # FP8 e4m3 (unit scales; the inputs are in [-1, 1]).
        a8 = a.astype(jnp.float8_e4m3fn)
        b8 = b.astype(jnp.float8_e4m3fn)
        bt8 = jax.block_until_ready(jax.jit(lambda y: y.T)(b8))  # w stored [N, K]
        ref8 = jnp.dot(a8.astype(jnp.float32), b8.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        f8_kn = jax.jit(lambda x, y: lax.dot_general(x, y, (((1,), (0,)), ((), ())),
                                                     preferred_element_type=jnp.bfloat16))
        f8_nk = jax.jit(lambda x, y: lax.dot_general(x, y, (((1,), (1,)), ((), ())),
                                                     preferred_element_type=jnp.bfloat16))
        rec["cublas_fp8_w_kn"] = _tflops(m, k, n, _time(f8_kn, a8, b8))
        rec["cublas_fp8_w_nk"] = _tflops(m, k, n, _time(f8_nk, a8, bt8))
        for key, fn, args in (("cublas_fp8_w_kn", f8_kn, (a8, b8)), ("cublas_fp8_w_nk", f8_nk, (a8, bt8))):
            if "median_ms" in rec[key]:
                rec[key]["rel_err"] = _err(fn(*args), ref8)
                hlo = fn.lower(*args).compile().as_text()
                rec[key]["cublaslt_f8_calls"] = hlo.count("__cublas$lt$matmul$f8")
                rec[key]["transposes"] = hlo.count(" transpose(")
                rec[key]["gemm_kinds"] = _gemm_kinds(hlo)
        mgpu8 = {}
        for cfg in _configs(128):
            mgpu8[_cfg_name(cfg)] = _run_mosaic(lambda c: functools.partial(fp8_matmul, config=c),
                                                cfg, (a8, bt8), ref8, (m, k, n))
        rec["mosaic_fp8"] = mgpu8
        out[name] = rec
        for key in ("cublas_bf16", "cublas_fp8_w_kn", "cublas_fp8_w_nk"):
            print(name, key, rec[key], flush=True)
        for group in ("mosaic_bf16", "mosaic_fp8"):
            for cname, v in rec[group].items():
                print(f"  {group} {cname}: {v.get('tflops')} TFLOP/s err={v.get('rel_err')} {v.get('error', '')}",
                      flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Programmatic dependent launch
# --------------------------------------------------------------------------------------------
def _chain(n_kernels: int, pdl: bool, size: int):
    def body(x_ref, o_ref):
        if pdl:
            plgpu.griddepcontrol_wait()
        o_ref[...] = x_ref[...] * 0.5 + 1.0
        if pdl:
            plgpu.griddepcontrol_launch_dependents()

    k = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((size,), jnp.float32))

    def f(x):
        for _ in range(n_kernels):
            x = k(x)
        return x

    return jax.jit(f)


def _wall_us(fn, x, reps: int) -> float:
    for _ in range(5):
        jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


_LAUNCH_RE = re.compile(r"call i32 @mosaic_gpu_launch_kernel\([^\n]*?, i32 (\d+), i32 (\d+), ptr %\w+, ptr %\w+\)")


def _first_run_with_host_llvm(fn, x):
    """Run ``fn`` once (its Mosaic GPU kernels compile now: this process has no compilation cache)
    with Mosaic GPU's host-LLVM dump on, and return (output, the ``uses_pdl`` launch argument of
    every ``mosaic_gpu_launch_kernel`` call in the dump). The dump goes to file descriptor 1."""
    key = "MOSAIC_GPU_DUMP_HOST_LLVM"
    old = os.environ.get(key)
    os.environ[key] = "1"
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"probe_host_llvm_{os.getpid()}.txt")
    try:
        with open(tmp, "w+b") as f:
            os.dup2(f.fileno(), 1)
            try:
                out = np.asarray(jax.block_until_ready(fn(x)))
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
            f.seek(0)
            text = f.read().decode(errors="replace")
    finally:
        os.close(saved)
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
        if os.path.exists(tmp):
            os.remove(tmp)
    return out, [int(m.group(2)) for m in _LAUNCH_RE.finditer(text)]


def pdl_probe(n_kernels: int = 64, size: int = 128, reps: int = 200) -> dict:
    x = jnp.arange(size, dtype=jnp.float32)
    res = dict(n_kernels=n_kernels, size=size, reps=reps)
    expect = np.asarray(x)
    for _ in range(n_kernels):
        expect = expect * 0.5 + 1.0
    for pdl in (False, True):
        tag = "pdl" if pdl else "no_pdl"
        try:
            one = _chain(1, pdl, size)
            many = _chain(n_kernels, pdl, size)
            got, launch_uses_pdl = _first_run_with_host_llvm(many, x)
            t1 = _wall_us(one, x, reps)
            tn = _wall_us(many, x, reps)
            hlo = many.lower(x).compile().as_text()
            res[tag] = dict(
                wall_us_1=round(t1, 1), wall_us_n=round(tn, 1),
                us_per_extra_kernel=round((tn - t1) / (n_kernels - 1), 2),
                max_abs_err=float(np.max(np.abs(got - expect))),
                # one entry per compiled Mosaic GPU module (the 64 calls share one kernel):
                # 1 = launched with programmatic stream serialization, [] = no dump seen
                launch_uses_pdl=launch_uses_pdl,
                custom_calls=hlo.count("custom_call_target=\"mosaic_gpu"),
                command_buffer=("command_buffer" in hlo),
                kernel_ms=harness.median_kernel_ms(many, x, iterations=10),
            )
        except Exception as e:
            res[tag] = dict(error=harness.error_summary(f"{type(e).__name__}: {e}"),
                            trace=traceback.format_exc()[-1500:])
        print(tag, res[tag], flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--shapes", default=",".join(s for s in SHAPES if s != "quick"))
    p.add_argument("--skip-gemm", action="store_true")
    p.add_argument("--skip-pdl", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="correctness only: one small shape, 3 timed calls (the times are not results)")
    args = p.parse_args()
    global ITERS
    # profiler.measure keeps every timed call's output alive: 10 calls of the 16384 x 16384 bf16
    # head tile already hold 5.4 GB, 30 calls ran out of memory at MEM=0.12.
    ITERS = 10
    if args.quick:
        ITERS = 3
        args.shapes = "quick"
    payload = dict(env=harness.environment(), gpu_before=harness.gpu_snapshot(), quick=args.quick)
    if not args.skip_pdl:
        payload["pdl"] = pdl_probe(reps=20 if args.quick else 200)
        harness.write_json(args.json, payload)
    if not args.skip_gemm:
        payload["gemm"] = gemm_probe([s for s in args.shapes.split(",") if s])
    payload["gpu_after"] = harness.gpu_snapshot()
    harness.write_json(args.json, payload)
    print("wrote", args.json)


if __name__ == "__main__":
    main()
