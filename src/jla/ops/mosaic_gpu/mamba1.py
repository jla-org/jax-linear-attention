# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), csrc/selective_scan/selective_scan_fwd_kernel.cuh,
# mamba_ssm/ops/selective_scan_interface.py
"""Mamba-1 selective scan on Mosaic GPU (H100): ``linear_attention("mamba1", impl="mosaic_gpu")``.

The semantics of :func:`jla.ops.reference.mamba1` (the official ``selective_scan_fn``):
``delta = softplus(dt + dt_bias)``, ``h_t = exp(delta_t A) h_{t-1} + delta_t u_t B_t``,
``y_t = C_t h_t + D u_t``, ``o_t = y_t silu(z_t)``, the state ``[B, D, N]`` in float32.

Design (the official CUDA kernel's, mapped to warpgroup arrays): one CTA per
``(block of 128 channels, batch)``, one channel per thread, the channel's ``N`` states in
registers. The CTA walks the sequence in TMA tiles of :data:`CHUNK` tokens (``dt, u, z`` as
``[CHUNK, 128]`` tiles, ``B, C`` as ``[CHUNK, N]`` tiles read as warp-uniform scalars), with a
:data:`STAGES`-deep prefetch. Each token fuses ``softplus(dt + bias)``, the ``N`` state updates
(``exp2`` of the pre-scaled ``A``), the ``C`` contraction, the ``D`` skip and the ``z`` gate; the
output row goes straight to GMEM (coalesced, no shared-memory round trip), so the only CTA-wide
synchronisation is one barrier per tile before its slot is refilled.

Three backward passes: ``backward="kernel"`` runs the Mosaic GPU backward kernel
(:mod:`jla.ops.mosaic_gpu.mamba1_bwd2`), for which the forward also saves the state entering every
tile; ``backward="kernel_v1"`` runs the first backward kernel (:mod:`jla.ops.mosaic_gpu.mamba1_bwd`, kept
for comparison); ``backward="reference"`` differentiates the reference op (an XLA token scan). The
default, ``"auto"``, takes the kernel for every ``d_state`` it supports.

``impl="auto"`` takes these kernels for the calls :func:`supports` accepts (an H100, ``D`` a multiple of
128, ``B, C`` shared by all channels, bf16 or f32, ``d_state`` 4, 8, 16 or 32 so that the backward
kernel runs): they are tested there against the reference, and they replace the chunked XLA op, which
is several times slower and whose gradient goes wrong on the H100 for some inputs
(``tests/test_mosaic_gpu_mamba1.py``).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla import ops
from jla.ops.reference import mamba1_inputs, segment_resets

__all__ = ["BLOCK_D", "CHUNK", "SUPPORTED_DSTATE", "mamba1", "mamba1_sharded", "supports"]

Array = jax.Array

CHUNK = 32  # tokens per TMA tile
BLOCK_D = 128  # channels per CTA: one per thread of the warpgroup
STAGES = 3  # TMA prefetch depth (tiles in flight)
UNROLL = 4  # tokens per iteration of the in-tile loop (a full unroll only slows compilation)
SUPPORTED_DSTATE = (4, 8, 16, 32)
KERNEL_BACKWARD_DSTATE = (4, 8, 16, 32)  # the backward kernel (32 as two 16-state slices)
KERNEL_V1_BACKWARD_DSTATE = (4, 8, 16)  # the first backward kernel (backward="kernel_v1")
SUPPORTED_DTYPES = (jnp.bfloat16, jnp.float32)  # the GPU-tested ones
_LOG2E = math.log2(math.e)
_PAD_DT = -1.0e4  # softplus(-1e4 + bias) == 0: padded tokens leave the state unchanged
_F32 = jnp.float32
_VEC = plgpu.Layout.WG_STRIDED((BLOCK_D,), vec_size=1)  # thread i holds channel i


@dataclasses.dataclass(frozen=True)
class Config:
    """Static shapes and options of one kernel call."""

    batch: int
    num_chunks: int
    dim: int
    dstate: int
    softplus: bool
    has_z: bool
    segments: bool
    save_states: bool = False  # also return the state entering every tile (for the backward kernel)

    @property
    def seq_len(self) -> int:
        return self.num_chunks * CHUNK


def _on_hopper() -> bool:
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and getattr(device, "compute_capability", "") == "9.0"


def _kernel(cfg: Config, name: str, body, **kwargs):
    # No automatic barriers: every thread touches only its own channel's column of the shared
    # tiles, and the one cross-thread hazard (refilling a tile slot) is fenced explicitly.
    return plgpu.kernel(
        body,
        grid=(cfg.dim // BLOCK_D, cfg.batch),
        grid_names=("d", "b"),
        kernel_name=name,
        compiler_params=plgpu.CompilerParams(approx_math=True, unsafe_no_auto_barriers=True),
        **kwargs,
    )


def _vec(ref) -> Array:
    return plgpu.load(ref, layout=_VEC, optimized=False)


def _softplus(x: Array) -> Array:
    """``log(1 + exp(x))`` (``x`` above 20 passes through, as in the official kernel)."""
    e = jnp.exp(jnp.minimum(x, 20.0))
    w = 1.0 + e
    # log1p(e) to a few ulp: log(w) * e / (w - 1), or e itself once 1 + e rounds to 1.
    lp = jnp.where(w == 1.0, e, jnp.log(w) * (e / (w - 1.0)))
    return jnp.where(x > 20.0, x, lp)


def _delta(cfg: Config, dt: Array, bias: Array) -> Array:
    x = dt.astype(_F32) + bias
    return _softplus(x) if cfg.softplus else x


def _tiles(cfg: Config, refs: dict, b, d0, n, slot, bar) -> None:
    """Issue the TMA loads of tile ``n`` into ``slot``."""
    rows = pl.ds(n * CHUNK, CHUNK)
    cols = pl.ds(d0, BLOCK_D)
    for name in ("dt", "u") + (("z",) if cfg.has_z else ()):
        plgpu.copy_gmem_to_smem(refs[name].at[b, rows, cols], refs[name + "_s"].at[slot], bar)
    for name in ("bm", "cm"):
        plgpu.copy_gmem_to_smem(refs[name].at[b, rows], refs[name + "_s"].at[slot], bar)
    if cfg.segments:
        plgpu.copy_gmem_to_smem(refs["keep"].at[b, rows], refs["keep_s"].at[slot], bar)


def _tile_smem(cfg: Config, dtypes: dict[str, Any]) -> dict[str, Any]:
    smem = {}
    for name in ("dt", "u") + (("z",) if cfg.has_z else ()):
        smem[name + "_s"] = plgpu.SMEM((STAGES, CHUNK, BLOCK_D), dtypes[name])
    for name in ("bm", "cm"):
        smem[name + "_s"] = plgpu.SMEM((STAGES, CHUNK, cfg.dstate), _F32)
    if cfg.segments:
        smem["keep_s"] = plgpu.SMEM((STAGES, CHUNK), _F32)
    return smem


def _num_tile_copies(cfg: Config) -> int:
    return 4 + int(cfg.has_z) + int(cfg.segments)


# ---------------------------------------------------------------------------------------------
# forward kernel
# ---------------------------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=0)  # one lowering per config, shared by every call site
def _forward_call(cfg: Config, dt, u, z, bm, cm, keep, bias, dskip, a2t, h0t):
    """Kernel inputs: ``dt, u, z [B, T, D]``; ``bm, cm [B, T, N]`` f32; ``keep [B, T]`` f32;
    ``bias, dskip [D]`` f32; ``a2t [N, D]`` f32 (``A * log2 e``, transposed); ``h0t [B, N, D]`` f32.
    Returns ``o [B, T, D]`` (u's dtype), ``final [B, N, D]`` (float32).

    Jitted with ``cfg`` static: the Mosaic GPU lowering of the kernels, not XLA's compile, is what
    takes long (about 20 s here and two minutes for the backward kernel, per configuration), and
    inside one traced program every call of a jitted function with the same configuration and
    shapes shares one lowering, so the Mamba-1 layers of a model that does not scan its layers
    (Jamba's interleaved stack) lower the kernels once instead of once per layer."""
    n_state, nc = cfg.dstate, cfg.num_chunks
    names = ["dt", "u"] + (["z"] if cfg.has_z else []) + ["bm", "cm"] + (["keep"] if cfg.segments else [])
    names += ["bias", "dskip", "a2t", "h0t"]
    arrays = dict(dt=dt, u=u, z=z, bm=bm, cm=cm, keep=keep, bias=bias, dskip=dskip, a2t=a2t, h0t=h0t)
    out_names = ["o", "final"] + (["states"] if cfg.save_states else [])
    out_types = [
        jax.ShapeDtypeStruct(u.shape, u.dtype),
        jax.ShapeDtypeStruct((cfg.batch, n_state, cfg.dim), _F32),
    ]
    if cfg.save_states:
        out_types.append(jax.ShapeDtypeStruct((cfg.batch, nc, n_state, cfg.dim), _F32))
    smem = _tile_smem(cfg, dict(dt=dt.dtype, u=u.dtype, z=None if z is None else z.dtype))
    smem["bars"] = plgpu.Barrier(num_arrivals=_num_tile_copies(cfg), num_barriers=STAGES)

    def body(*refs_in, **scratch):
        refs = dict(zip(names + out_names, refs_in))
        refs.update(scratch)
        bars = scratch["bars"]
        di, b = lax.axis_index("d"), lax.axis_index("b")
        d0 = di * BLOCK_D
        cols = pl.ds(d0, BLOCK_D)
        for i in range(min(STAGES, nc)):
            _tiles(cfg, refs, b, d0, i, i, bars.at[i])
        bias_v = _vec(refs["bias"].at[cols])
        dskip_v = _vec(refs["dskip"].at[cols])
        a2 = [_vec(refs["a2t"].at[j, cols]) for j in range(n_state)]
        h_init = tuple(_vec(refs["h0t"].at[b, j, cols]) for j in range(n_state))

        def tile(n, h):
            slot = lax.rem(n, STAGES)
            if cfg.save_states:
                for j in range(n_state):
                    refs["states"][b, n, j, cols] = h[j]
            plgpu.barrier_wait(bars.at[slot])
            dt_s, u_s, bm_s, cm_s = (refs[k].at[slot] for k in ("dt_s", "u_s", "bm_s", "cm_s"))

            def token(t, h):
                delta = _delta(cfg, _vec(dt_s.at[t]), bias_v)
                uu = _vec(u_s.at[t]).astype(_F32)
                du = delta * uu
                if cfg.segments:
                    keep_t = refs["keep_s"][slot, t]
                y = dskip_v * uu
                for j in range(n_state):
                    decay = jnp.exp2(delta * a2[j])
                    if cfg.segments:
                        decay = decay * keep_t
                    h[j] = decay * h[j] + du * bm_s[t, j]
                    y = y + h[j] * cm_s[t, j]
                if cfg.has_z:
                    zz = _vec(refs["z_s"].at[slot, t]).astype(_F32)
                    y = y * (zz * jax.nn.sigmoid(zz))
                refs["o"][b, n * CHUNK + t, cols] = y.astype(refs["o"].dtype)
                return h

            def group(i, h):  # UNROLL tokens per loop iteration
                h = list(h)
                for r in range(UNROLL):
                    h = token(i * UNROLL + r, h)
                return tuple(h)

            h = lax.fori_loop(0, CHUNK // UNROLL, group, h)
            # Every thread is done reading this slot: refill it with tile n + STAGES.
            plgpu.commit_smem()

            @pl.when(n + STAGES < nc)
            def _():
                _tiles(cfg, refs, b, d0, n + STAGES, slot, bars.at[slot])

            return tuple(h)

        h_final = lax.fori_loop(0, nc, tile, h_init)
        for j in range(n_state):
            refs["final"][b, j, cols] = h_final[j]

    call = _kernel(cfg, "mamba1_fwd", body, out_type=tuple(out_types), scratch_types=smem)
    return call(*(arrays[k] for k in names))


# ---------------------------------------------------------------------------------------------
# wrapper and custom VJP
# ---------------------------------------------------------------------------------------------


def _prepare(cfg: Config, seq_len: int, dt, u, z, bm, cm, keep, bias, dskip, a, h0):
    """Pad to whole tiles and lay out the kernel inputs (see :func:`_forward_call`)."""
    pad = cfg.seq_len - seq_len

    def tpad(x, value=0.0):
        return x if pad == 0 else jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2), constant_values=value)

    dt_p = tpad(dt, _PAD_DT)
    u_p, z_p = tpad(u), None if z is None else tpad(z)
    bm_p, cm_p = tpad(bm.astype(_F32)), tpad(cm.astype(_F32))
    keep_p = None if keep is None else tpad(keep, 1.0)
    a2t = (a.astype(_F32) * _LOG2E).T
    h0t = jnp.swapaxes(h0.astype(_F32), 1, 2)
    return dt_p, u_p, z_p, bm_p, cm_p, keep_p, bias.astype(_F32), dskip.astype(_F32), a2t, h0t


def _forward(cfg: Config, seq_len: int, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids):
    keep = None if segment_ids is None else jnp.where(segment_resets(segment_ids), 0.0, 1.0).astype(_F32)
    prepared = _prepare(cfg, seq_len, dt, v, z, k, q, keep, dt_bias, d, a, h0)
    o, final = _forward_call(cfg, *prepared)
    return o[:, :seq_len], jnp.swapaxes(final, 1, 2)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _scan(cfg: Config, seq_len: int, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids):
    return _forward(cfg, seq_len, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids)


def _scan_fwd(cfg: Config, seq_len: int, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids):
    out = _forward(cfg, seq_len, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids)
    return out, (q, k, v, dt, a, d, z, dt_bias, h0, segment_ids)


def _scan_bwd(cfg: Config, seq_len: int, res, cts):
    """The backward pass of :func:`jla.ops.reference.mamba1` (XLA token scan).

    Not the chunked op's: on the H100 its gradient (``lax.associative_scan`` under
    ``jax.checkpoint``) is wrong for some inputs, see ``tests/test_mosaic_gpu_mamba1.py``.
    """
    q, k, v, dt, a, d, z, dt_bias, h0, segment_ids = res

    def scan(q, k, v, dt, a, d, z, dt_bias, h0):
        return ops.reference.mamba1(q, k, v, dt, a, d=d, z=z, dt_bias=dt_bias, dt_softplus=cfg.softplus,
                                    initial_state=h0, output_final_state=True, segment_ids=segment_ids)

    _, vjp = jax.vjp(scan, q, k, v, dt, a, d, z, dt_bias, h0)
    return (*vjp(cts), None)


_scan.defvjp(_scan_fwd, _scan_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _scan_kernel_bwd(cfg: Config, seq_len: int, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids):
    return _forward(cfg, seq_len, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids)


def _scan_kernel_bwd_fwd(cfg: Config, seq_len: int, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids):
    keep = None if segment_ids is None else jnp.where(segment_resets(segment_ids), 0.0, 1.0).astype(_F32)
    scfg = dataclasses.replace(cfg, save_states=True)
    prepared = _prepare(scfg, seq_len, dt, v, z, k, q, keep, dt_bias, d, a, h0)
    o, final, states = _forward_call(scfg, *prepared)
    out = (o[:, :seq_len], jnp.swapaxes(final, 1, 2))
    return out, (prepared, states, q, k, v, dt, a, d, z, dt_bias, h0)


def _scan_kernel_bwd_bwd(cfg: Config, seq_len: int, res, cts):
    from jla.ops.mosaic_gpu.mamba1_bwd2 import backward_call

    return _kernel_backward(cfg, seq_len, res, cts, backward_call)


def _scan_kernel_v1_bwd(cfg: Config, seq_len: int, res, cts):
    from jla.ops.mosaic_gpu.mamba1_bwd import backward_call

    return _kernel_backward(cfg, seq_len, res, cts, backward_call)


def _kernel_backward(cfg: Config, seq_len: int, res, cts, backward_call):
    prepared, states, q, k, v, dt, a, d, z, dt_bias, h0 = res
    dt_p, u_p, z_p, bm_p, cm_p, keep_p, bias, dskip, a2t, _ = prepared
    do, dfinal = cts
    pad = cfg.seq_len - seq_len
    do_p = do if pad == 0 else jnp.pad(do, [(0, 0), (0, pad), (0, 0)])
    ddt, du, dz, dbc, ddskip, dbias, da, dh0 = backward_call(
        cfg, dt_p, u_p, z_p, do_p.astype(u_p.dtype), bm_p, cm_p, keep_p, bias, dskip, a2t, states,
        jnp.swapaxes(dfinal.astype(_F32), 1, 2))
    n = cfg.dstate
    dbc = jnp.sum(dbc, axis=1).reshape(cfg.batch, cfg.seq_len, 2 * n)[:, :seq_len]
    sl = slice(0, seq_len)
    return (dbc[..., :n].astype(q.dtype), dbc[..., n:].astype(k.dtype), du[:, sl].astype(v.dtype),
            ddt[:, sl].astype(dt.dtype), jnp.sum(da, axis=0).T.astype(a.dtype),
            jnp.sum(ddskip, axis=0).astype(d.dtype), None if z is None else dz[:, sl].astype(z.dtype),
            jnp.sum(dbias, axis=0).astype(dt_bias.dtype), jnp.swapaxes(dh0, 1, 2).astype(h0.dtype), None)


_scan_kernel_bwd.defvjp(_scan_kernel_bwd_fwd, _scan_kernel_bwd_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _scan_kernel_v1(cfg: Config, seq_len: int, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids):
    return _forward(cfg, seq_len, q, k, v, dt, a, d, z, dt_bias, h0, segment_ids)


_scan_kernel_v1.defvjp(_scan_kernel_bwd_fwd, _scan_kernel_v1_bwd)


# ---------------------------------------------------------------------------------------------
# public op
# ---------------------------------------------------------------------------------------------


def _unsupported(q, k, v, dt, a, *, d=None, z=None, dt_bias=None, dt_softplus=False, scale=None,
                 initial_state=None, output_final_state=False, cu_seqlens=None, segment_ids=None,
                 chunk_size=None, backward="auto") -> str | None:
    """Why the kernels cannot run this call (None if they can)."""
    del d, dt_bias, initial_state, output_final_state, chunk_size
    if backward not in ("auto", "kernel", "kernel_v1", "reference"):
        return f"backward={backward!r} (auto, kernel, kernel_v1 or reference)"
    if backward == "kernel" and a.ndim == 2 and a.shape[1] not in KERNEL_BACKWARD_DSTATE:
        return f"the backward kernel with d_state = {a.shape[1]} (needs one of {KERNEL_BACKWARD_DSTATE})"
    if backward == "kernel_v1" and a.ndim == 2 and a.shape[1] not in KERNEL_V1_BACKWARD_DSTATE:
        return f"the first backward kernel with d_state = {a.shape[1]} (needs one of {KERNEL_V1_BACKWARD_DSTATE})"
    if cu_seqlens is not None:
        return "cu_seqlens (pass segment_ids)"
    if scale is not None and scale != 1.0:
        return "an output scale"
    if v.ndim != 3:
        return f"v of shape {v.shape}"
    batch, seq_len, dim = v.shape
    if dim % BLOCK_D:
        return f"D = {dim} (needs a multiple of {BLOCK_D})"
    if a.ndim != 2 or a.shape[1] not in SUPPORTED_DSTATE:
        return f"A of shape {a.shape} (d_state in {SUPPORTED_DSTATE})"
    if q.shape != (batch, seq_len, a.shape[1]) or k.shape != q.shape:
        return "grouped or per-channel B/C (only [B, T, N])"
    if any(x.dtype not in SUPPORTED_DTYPES for x in (v, dt) + (() if z is None else (z,))):
        return "u, dt, z dtypes other than bf16 or f32"
    if dt.dtype != v.dtype or (z is not None and z.dtype != v.dtype):
        return "u, dt and z of different dtypes"
    if not dt_softplus and seq_len % CHUNK:
        return f"dt_softplus=False with T % {CHUNK} != 0 (padding needs softplus)"
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        return "segment_ids of the wrong shape"
    return None


def supports(q, k, v, dt, a, **options) -> bool:
    """True if :func:`mamba1` runs this call on this machine (an sm_90 GPU) with the backward kernel, or with
    the reference backward asked for by name (``backward="reference"``)."""
    if not _on_hopper() or _unsupported(q, k, v, dt, a, **options) is not None:
        return False
    return options.get("backward", "auto") == "reference" or a.shape[1] in KERNEL_BACKWARD_DSTATE


def mamba1(
    q: Array,
    k: Array,
    v: Array,
    dt: Array,
    a: Array,
    *,
    d: Array | None = None,
    z: Array | None = None,
    dt_bias: Array | None = None,
    dt_softplus: bool = False,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int | None = None,
    backward: str = "auto",
) -> tuple[Array, Array | None]:
    """Mamba-1 selective scan (:func:`jla.ops.reference.mamba1`) with the Mosaic GPU forward kernel.

    Differentiable in ``q (C), k (B), v (u), dt, a, d, z, dt_bias, initial_state``. ``backward``:
    ``"kernel"`` (the Mosaic GPU backward kernel of :mod:`jla.ops.mosaic_gpu.mamba1_bwd2`; d_state in
    :data:`KERNEL_BACKWARD_DSTATE`), ``"kernel_v1"`` (the first backward kernel, :mod:`jla.ops.mosaic_gpu.mamba1_bwd`,
    kept for comparison; d_state in :data:`KERNEL_V1_BACKWARD_DSTATE`), ``"reference"`` (the reference op's VJP,
    XLA) or ``"auto"`` (the kernel where it applies).
    ``chunk_size`` is accepted for interface compatibility (the kernels tile by :data:`CHUNK`).
    """
    reason = _unsupported(q, k, v, dt, a, d=d, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus, scale=scale,
                          initial_state=initial_state, cu_seqlens=cu_seqlens, segment_ids=segment_ids,
                          backward=backward)
    if reason is not None:
        raise ValueError(f"the mamba1 Mosaic GPU kernel does not support {reason}")
    mamba1_inputs(q, k, v, dt, a, dt_bias=dt_bias, dt_softplus=dt_softplus, scale=scale, cu_seqlens=cu_seqlens,
                  segment_ids=segment_ids, initial_state=initial_state)  # shape checks
    del chunk_size
    batch, seq_len, dim = v.shape
    dstate = a.shape[1]
    cfg = Config(batch=batch, num_chunks=-(-seq_len // CHUNK), dim=dim, dstate=dstate, softplus=dt_softplus,
                 has_z=z is not None, segments=segment_ids is not None)
    dskip = jnp.zeros((dim,), _F32) if d is None else d
    bias = jnp.zeros((dim,), _F32) if dt_bias is None else dt_bias
    h0 = jnp.zeros((batch, dim, dstate), _F32) if initial_state is None else initial_state
    if backward == "auto":
        backward = "kernel" if dstate in KERNEL_BACKWARD_DSTATE else "reference"
    scan = dict(reference=_scan, kernel=_scan_kernel_bwd, kernel_v1=_scan_kernel_v1)[backward]
    o, final = scan(cfg, seq_len, q, k, v, dt, a, dskip, z, bias, h0, segment_ids)
    return o, (final if output_final_state else None)


#: Layouts of the op's array arguments for :func:`jla.distributed.kernel_sharding.shard_layout_call`: the
#: channels ``D`` play the heads (Mamba-1 has none); ``B`` and ``C`` are shared by every channel.
_LAYOUTS = dict(q="batch", k="batch", v="token", dt="token", a="channels", d="channels", z="token",
                dt_bias="channels", initial_state="state", segment_ids="batch")


def mamba1_sharded(q: Array, k: Array, v: Array, dt: Array, a: Array, *, d: Array | None = None,
                   z: Array | None = None, dt_bias: Array | None = None, initial_state: Array | None = None,
                   segment_ids: Array | None = None, **static: Any) -> tuple[Array, Array | None]:
    """:func:`mamba1` per shard of the context mesh (the dispatcher's ``impl="mosaic_gpu"`` entry).

    The kernels are custom calls that XLA cannot partition, and ``a [D, N]`` is not laid out like the
    linear-attention kinds' gates, so the dispatcher's own split skips ``mamba1``. Here the batch is split
    over the ``batch`` rule's axes and the channels over the ``heads`` rule's axes
    (:func:`jla.distributed.kernel_sharding.shard_layout_call`): ``u, dt, z`` and the state by both, ``B, C``
    and ``segment_ids`` by the batch only (each channel shard's ``dB, dC`` are partial sums over its
    channels, which ``shard_map``'s transpose adds up), ``A, D, dt_bias`` with the channels. Each split is
    checked on the per-device shapes (``D`` per device must stay a multiple of 128); without a split that
    the kernels take, the call runs unsplit. Without a context mesh this is :func:`mamba1`.
    """
    from jla.distributed.kernel_sharding import shard_layout_call

    def call(q, k, v, dt, a, d, z, dt_bias, initial_state, segment_ids):
        return mamba1(q, k, v, dt, a, d=d, z=z, dt_bias=dt_bias, initial_state=initial_state,
                      segment_ids=segment_ids, **static)

    def local_ok(q, k, v, dt, a, d, z, dt_bias, initial_state, segment_ids) -> bool:
        return _unsupported(q, k, v, dt, a, d=d, z=z, dt_bias=dt_bias, initial_state=initial_state,
                            segment_ids=segment_ids, **static) is None

    operands = dict(q=q, k=k, v=v, dt=dt, a=a, d=d, z=z, dt_bias=dt_bias, initial_state=initial_state,
                    segment_ids=segment_ids)
    return shard_layout_call(call, operands, _LAYOUTS, out_layouts=("token", "state"), heads=v.shape[-1],
                             supports=local_ok)


ops.register("mosaic_gpu", "mamba1", mamba1_sharded, supports, auto=True)
