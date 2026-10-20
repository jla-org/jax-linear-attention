# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run single-device kernels on the local shards of sharded activations.

Pallas Mosaic GPU kernels lower to opaque custom calls that XLA's SPMD
partitioner cannot split. Under ``jax.jit`` with batch- or head-sharded inputs
it all-gathers every operand and runs the whole kernel on every device: the
result is right, but the kernel gets no parallel speedup and every device holds
the full activations (on 2 H100s the batch-sharded Gated DeltaNet
forward+backward ran 1.6x slower than on one GPU; see ``benchmarks/multi_gpu.md``).

:func:`shard_kernel_call` runs the kernel inside ``jax.shard_map`` over the
context mesh instead (``jax.set_mesh``; :class:`jla.training.trainer.Trainer`
sets it while it traces its steps, together with its sharding rules through
:func:`use_rules`). The batch is split over the mesh axes of the ``batch`` rule
and the heads over the axes of the ``heads`` rule (``("dp", "fsdp")`` and
``("tp",)`` by default), so each device runs the kernel on its own rows and
heads, as one device would. Every other mesh axis of size > 1 (one the rules do
not place, or that does not divide its dimension) splits the batch too, or else
the heads: a split never leaves a mesh axis out, since an axis left out runs the
kernel repeated on every device of it and its ``shard_map`` transpose was not
exact (see :func:`_splits`). Axes of size 1 and axes that are already manual
(inside another ``shard_map``) are skipped.

The kernel's ``supports`` predicate is checked again on the per-device shapes:
a kernel that accepts the global call may reject a shard (e.g. ``H * K`` must be
a multiple of 128 for the vector-gate kernels). The split then falls back to
the batch only (all axes on the batch), then to the heads only, then to the
unsplit call (gathered inputs, the whole kernel on every device), which is what
ran before.

The kernel is called unchanged when there is no context mesh, when no axis
applies, with ``cu_seqlens`` (packed rows), for kinds not in
:data:`TOKEN_LAYOUT_KINDS` (gates that are parameters, such as Mamba-1's
``a [D, N]`` or RWKV-6's ``u [H, K]``, or layouts not checked yet, such as Gated
DeltaProduct's ``k [B, T * n, H, K]``), or when an input is not laid out as the
linear-attention ops document: ``q``/``k``/``v`` and gates ``[B, T, H, ...]``,
``initial_state`` (array or pytree) ``[B, H, ...]``, ``segment_ids`` ``[B, T]``. Splitting the batch is exact
because every op treats examples independently; heads are split only when every
head-carrying input and output has the same ``H`` (the layers repeat grouped key
heads before the call).

:func:`shard_layout_call` does the same for kernels that layers call directly, with
a layout per operand and replicated parameters: the FA3-style attention kernel
(``jla.layers.attention``, batch and GQA head blocks) and the fused Mamba-3 SISO core
(``jla.layers.ssm``, batch only, so under ``tp`` its batch is split over ``tp`` as well;
its bias and skip parameters enter replicated).

The other single-device kernels get rules in the same terms (each falls back to its
plain-JAX path when no split gives per-device shapes it takes, through ``unsplit``):
the fused short conv and the fused conv + q/k L2 norm (``jla.ops.mosaic_gpu.short_conv``,
``short_conv_l2norm``: the batch over the ``batch`` axes, the channels, with the conv weight
and bias, over the axes that shard the conv's channels), the FP8 fused SwiGLU passes
(``jla.layers.fp8_swiglu``: tokens over the batch axes, ``d_ff`` over the ``mlp`` axes, the
transposed FP8 copy as layout ``(1, 0)`` and the amax as one partial per shard,
``"shards"``) and the fused SwiGLU GEMMs (``jla.ops.mosaic_gpu.fused_gemm``: tokens over the
batch axes, weights replicated; under ``tp``, which shards the MLP's features, the XLA path
keeps that layout instead).
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
from jax.sharding import PartitionSpec as P

from jla.distributed.sharding import DEFAULT_RULES

__all__ = ["TOKEN_LAYOUT_KINDS", "kernel_shard_axes", "shard_kernel_call", "shard_layout_call", "use_rules",
           "mesh_can_split", "per_device_call", "per_device_step", "in_per_device_step", "rule_axes"]

Array = jax.Array

#: Array options the linear-attention ops accept, with known layouts.
_ARRAY_OPTIONS = frozenset({"initial_state", "segment_ids"})

#: Kinds whose gates are all per-token activations ``[B, T, H, ...]`` (see ``jla.ops.KINDS``). Calls
#: of other kinds, and of kinds registered later, run unsplit until they are added here. (``mlstm`` is
#: left out: its Mosaic GPU path splits only its simple-GLA kernel call, see ``jla.ops.mosaic_gpu.mlstm``.)
TOKEN_LAYOUT_KINDS = frozenset(
    {"simple_gla", "gla", "deltanet", "gated_deltanet", "kda", "gdn2", "mamba3", "mamba3_mimo", "rwkv7"}
)

_RULES: contextvars.ContextVar[Mapping[str, Sequence[str]]] = contextvars.ContextVar(
    "jla_kernel_sharding_rules", default=DEFAULT_RULES
)
_PER_DEVICE_STEP: contextvars.ContextVar[bool] = contextvars.ContextVar("jla_per_device_step", default=False)


@contextlib.contextmanager
def use_rules(rules: Mapping[str, Sequence[str]]) -> Iterator[None]:
    """Split kernels over the ``batch`` and ``heads`` axes of ``rules`` (the trainer's resolved rules)."""
    token = _RULES.set(rules)
    try:
        yield
    finally:
        _RULES.reset(token)


def _pick(candidates: Sequence[str], sizes: Mapping[str, int], dim: int, used: set[str]) -> tuple[str, ...]:
    chosen: list[str] = []
    product = 1
    for axis in candidates:
        size = sizes.get(axis, 1)
        if size == 1 or axis in used or dim % (product * size):
            continue
        chosen.append(axis)
        product *= size
    used.update(chosen)
    return tuple(chosen)


def kernel_shard_axes(
    mesh: Any,
    batch: int,
    heads: int | None,
    *,
    batch_axes: Sequence[str] | None = None,
    head_axes: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Non-manual mesh axes (size > 1) that split ``batch`` and ``heads`` (``None``: keep heads whole).

    ``batch_axes`` / ``head_axes`` default to the ``batch`` / ``heads`` rules set by :func:`use_rules`.
    """
    if mesh is None or mesh.empty:
        return (), ()
    rules = _RULES.get()
    batch_axes = tuple(rules.get("batch", ())) if batch_axes is None else batch_axes
    head_axes = tuple(rules.get("heads", ())) if head_axes is None else head_axes
    manual = set(getattr(mesh, "manual_axes", ()))
    sizes = {name: int(size) for name, size in zip(mesh.axis_names, mesh.axis_sizes) if name not in manual}
    used: set[str] = set()
    b_axes = _pick(batch_axes, sizes, batch, used)
    h_axes = _pick(head_axes, sizes, heads, used) if heads is not None else ()
    return b_axes, h_axes


def _splits(
    mesh: Any,
    batch: int,
    heads: int | None,
    batch_axes: Sequence[str] | None = None,
    head_axes: Sequence[str] | None = None,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Candidate ``(batch axes, head axes)`` splits, best first; each one splits a dim over EVERY
    non-manual mesh axis of size > 1 (``[]`` when there is none, or no split divides).

    ``shard_map`` treats an axis left out of the specs as replicated: every device of that axis runs the
    whole kernel, and (with ``check_vma=False``) the transpose divides the output cotangents by the axis
    size and sums the input cotangents over it. Besides repeating the work, that was not exact on H100:
    the fused Mamba-3 core under ``dp=2 x tp=2`` with the batch split over ``dp`` only gave every gradient
    off by up to 1.7e-2 of its maximum, while the same 4 rows per device under ``dp=2`` were bitwise equal
    to one GPU (``benchmarks/multi_gpu.md``). So an axis that the ``batch``/``heads`` rules do not place
    (``tp`` for a batch-only kernel, ``ep``, ``pp``, ...) splits the batch too, or else the heads.
    """
    if mesh is None or mesh.empty:
        return []
    manual = set(getattr(mesh, "manual_axes", ()))
    sizes = {name: int(size) for name, size in zip(mesh.axis_names, mesh.axis_sizes) if name not in manual}
    need = tuple(name for name in mesh.axis_names if sizes.get(name, 1) > 1)
    if not need:
        return []
    b_axes, h_axes = kernel_shard_axes(mesh, batch, heads, batch_axes=batch_axes, head_axes=head_axes)
    rest = tuple(axis for axis in need if axis not in b_axes and axis not in h_axes)

    def product(axes: tuple[str, ...]) -> int:
        out = 1
        for axis in axes:
            out *= sizes[axis]
        return out

    def valid(b: tuple[str, ...], h: tuple[str, ...]) -> bool:
        return (
            sorted((*b, *h)) == sorted(need)
            and batch % product(b) == 0
            and (not h or (heads is not None and heads % product(h) == 0))
        )

    candidates = (
        ((*b_axes, *rest), h_axes),  # the rules' split, other axes on the batch
        (b_axes, (*h_axes, *rest)),  # ... or on the heads
        ((*b_axes, *(axis for axis in need if axis not in b_axes)), ()),  # the batch only
        ((), (*h_axes, *(axis for axis in need if axis not in h_axes))),  # the heads only
    )
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for b, h in candidates:
        if valid(b, h) and (b, h) not in out:
            out.append((b, h))
    return out


def _manual_kwargs(mesh: Any) -> dict[str, Any]:
    """Inside another ``shard_map`` only the still-automatic axes can become manual."""
    manual = set(getattr(mesh, "manual_axes", ()))
    return {"axis_names": frozenset(mesh.axis_names) - manual} if manual else {}


def _entry(axes: tuple[str, ...]) -> Any:
    return None if not axes else axes[0] if len(axes) == 1 else axes


def _has_array(tree: Any) -> bool:
    return any(hasattr(leaf, "shape") and hasattr(leaf, "dtype") for leaf in jax.tree.leaves(tree))


def shard_kernel_call(
    fn: Callable[..., tuple[Array, Any]],
    q: Array,
    k: Array,
    v: Array,
    gates: Sequence[Array],
    options: Mapping[str, Any],
    *,
    supports: Callable[..., bool] | None = None,
    kind: str | None = None,
    batch_axes: Sequence[str] | None = None,
    head_axes: Sequence[str] | None = None,
) -> tuple[Array, Any]:
    """``fn(q, k, v, *gates, **options)`` on per-device shards of the context mesh (see the module docs).

    ``supports(q, k, v, *gates, **options)`` (the kernel's predicate) is evaluated on the local shapes of
    each candidate split; ``kind`` (a ``jla.ops.KINDS`` name) must be in :data:`TOKEN_LAYOUT_KINDS` for any
    split (``None``: the caller guarantees that every gate is laid out ``[B, T, H, ...]``).
    """
    gates = tuple(gates)
    mesh = jax.sharding.get_abstract_mesh()
    arrays = {name: value for name, value in options.items() if name in _ARRAY_OPTIONS and value is not None}
    static = {name: value for name, value in options.items() if name not in arrays}

    def body(q: Array, k: Array, v: Array, gates: tuple[Array, ...], arrays: dict[str, Any]) -> tuple[Array, Any]:
        return fn(q, k, v, *gates, **static, **arrays)

    def direct() -> tuple[Array, Any]:
        return body(q, k, v, gates, arrays)

    if (
        mesh.empty
        or (kind is not None and kind not in TOKEN_LAYOUT_KINDS)
        or options.get("cu_seqlens") is not None
        or q.ndim < 3
        or _has_array(static)
    ):
        return direct()
    batch, seq, heads = q.shape[:3]
    activations = (q, k, v, *gates)
    state_leaves = jax.tree.leaves(arrays.get("initial_state"))
    segment_ids = arrays.get("segment_ids")
    if (
        any(x.ndim < 2 or x.shape[:2] != (batch, seq) for x in activations)
        or any(x.ndim < 1 or x.shape[0] != batch for x in state_leaves)
        or (segment_ids is not None and tuple(segment_ids.shape) != (batch, seq))
        or not _splits(mesh, batch, heads, batch_axes, head_axes)
    ):
        return direct()
    o_shape, final_shape = jax.eval_shape(body, q, k, v, gates, arrays)
    final_leaves = jax.tree.leaves(final_shape)
    if o_shape.ndim < 3 or o_shape.shape[:2] != (batch, seq) or any(x.ndim < 1 or x.shape[0] != batch for x in final_leaves):
        return direct()
    head_split = (
        all(x.ndim >= 3 and x.shape[2] == heads for x in activations)
        and o_shape.shape[2] == heads
        and all(x.ndim >= 2 and x.shape[1] == heads for x in (*state_leaves, *final_leaves))
    )
    splits = _splits(mesh, batch, heads if head_split else None, batch_axes, head_axes)
    sizes = {name: int(size) for name, size in zip(mesh.axis_names, mesh.axis_sizes)}

    def local(x: Any, entries: tuple[Any, ...]) -> jax.ShapeDtypeStruct:
        """The per-device shape of ``x`` when its leading dims are split as ``entries``."""
        shape = list(x.shape)
        for dim, entry in enumerate(entries[: len(shape)]):
            for axis in () if entry is None else (entry,) if isinstance(entry, str) else entry:
                shape[dim] //= sizes[axis]
        return jax.ShapeDtypeStruct(tuple(shape), x.dtype)

    def accepted(b: Any, h: Any) -> bool:
        if supports is None:
            return True
        local_arrays = {
            name: local(value, (b,)) if name == "segment_ids" else jax.tree.map(lambda x: local(x, (b, h)), value)
            for name, value in arrays.items()
        }
        return bool(supports(*(local(x, (b, None, h)) for x in activations), **static, **local_arrays))

    # Splits that cover every mesh axis (see _splits), best first; else the unsplit call.
    candidates = ((_entry(b), _entry(h)) for b, h in splits)
    chosen = next((pair for pair in candidates if accepted(*pair)), None)
    if chosen is None:
        return direct()
    b, h = chosen

    def token(x: Any) -> P:  # [B, T, H, ...]
        return P(*(b, None, h)[: x.ndim])

    def state(x: Any) -> P:  # [B, H, ...]
        return P(*(b, h)[: x.ndim])

    array_specs = {
        name: P(b) if name == "segment_ids" else jax.tree.map(state, value) for name, value in arrays.items()
    }
    mapped = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(token(q), token(k), token(v), tuple(token(x) for x in gates), array_specs),
        out_specs=(token(o_shape), jax.tree.map(state, final_shape)),
        check_vma=False,
        **_manual_kwargs(mesh),
    )
    return mapped(q, k, v, gates, arrays)


#: Operand layouts of :func:`shard_layout_call`: the dims split by the batch and the head axes.
_LAYOUT_DIMS = {"token": (0, 2), "state": (0, 1), "batch": (0, None), "replicated": (None, None),
                "channels": (None, 0)}

#: Output layout of :func:`shard_layout_call` for per-shard partial results (e.g. an amax): each device's
#: ``[k, ...]`` block, stacked along dim 0 over every split mesh axis, so the global result is
#: ``[k * devices, ...]`` and the caller reduces it.
SHARDS = "shards"


def _is_dims(layout: Any) -> bool:
    return (isinstance(layout, tuple) and len(layout) == 2
            and all(d is None or (isinstance(d, int) and not isinstance(d, bool) and d >= 0) for d in layout))


def _layout_dims(layout: Any) -> tuple[int | None, int | None] | None:
    """``(batch dim, head dim)`` of a layout name or ``(batch_dim, head_dim)`` tuple (``None``: unknown)."""
    if isinstance(layout, str):
        return _LAYOUT_DIMS.get(layout)
    return layout if _is_dims(layout) else None


def shard_layout_call(
    fn: Callable[..., Any],
    operands: Mapping[str, Any],
    layouts: Mapping[str, Any],
    *,
    out_layouts: Any,
    heads: int | None = None,
    supports: Callable[..., bool] | None = None,
    head_axes: Sequence[str] | None = None,
    unsplit: Callable[..., Any] | None = None,
) -> Any:
    """``fn(**operands)`` per shard of the context mesh, for kernels called outside the dispatcher.

    Each operand (an array, a pytree of arrays, or ``None``) has a layout in ``layouts``:
    ``"token"`` ``[B, T, H, ...]``, ``"state"`` ``[B, H, ...]``, ``"batch"`` ``[B, ...]`` (heads never
    split), ``"channels"`` ``[H, ...]`` (a parameter split with the heads, e.g. a depthwise conv's
    ``[C, K]`` weight: replicated over the batch axes, whose devices' gradients ``shard_map``'s transpose
    sums), ``"replicated"`` (parameters: every device gets the whole array, and ``shard_map``'s transpose
    sums their gradients over the mesh), or a ``(batch_dim, head_dim)`` tuple for any other placement
    (either may be ``None``; e.g. ``(1, 0)`` for a transposed ``[H, B]`` copy). ``out_layouts`` is the
    layout of every output leaf: one layout, or one per element of a tuple output. An output needs a
    batch dim, except :data:`SHARDS`: one partial result per device, stacked along dim 0.
    The batch is split over the ``batch`` rule's axes and, when ``heads`` is given, the head dims over
    the ``heads`` rule's axes (``head_axes`` instead, when given) that divide ``heads`` (pass the gcd of
    the operands' head counts, e.g. of ``num_heads`` and ``num_kv_heads``: contiguous head blocks keep
    their GQA groups); every other mesh axis of size > 1 splits the batch as well (or else the heads),
    so no device repeats another's work (see :func:`_splits`).

    ``supports(**local_operands)`` (``jax.ShapeDtypeStruct`` leaves) is checked on the per-device shapes
    of each candidate split, with the fallbacks of :func:`shard_kernel_call`: batch and heads, the batch
    only, the heads only (each over every mesh axis), then the unsplit call: ``fn`` on the whole operands
    (gathered inputs), or ``unsplit(**operands)`` when given (e.g. the kernel's plain-JAX path, which XLA
    partitions; it may also return ``None`` for the caller to handle).
    """
    unknown = sorted(set(operands) - set(layouts))
    bad = sorted(name for name, layout in layouts.items() if _layout_dims(layout) is None)
    if unknown or bad:
        raise ValueError(f"shard_layout_call: operands without a layout {unknown}; unknown layouts {bad}")
    present = {name: value for name, value in operands.items() if value is not None}
    missing = {name: None for name in operands if name not in present}

    def body(arrays: dict[str, Any]) -> Any:
        return fn(**arrays, **missing)

    def fallback() -> Any:
        return body(present) if unsplit is None else unsplit(**present, **missing)

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return body(present)
    flat_in, in_tree = jax.tree.flatten(present)
    # One layout per leaf, in the flattening order of ``present`` (sorted keys); tuple layouts stay whole.
    in_layouts = [layouts[name] for name in sorted(present) for _ in jax.tree.leaves(present[name])]
    if len(in_layouts) != len(flat_in) or any(not hasattr(x, "shape") or not hasattr(x, "dtype") for x in flat_in):
        return fallback()
    rows = [(x, _layout_dims(layout)[0]) for layout, x in zip(in_layouts, flat_in, strict=True)]
    rows = [(x, dim) for x, dim in rows if dim is not None]
    if not rows or any(x.ndim <= dim for x, dim in rows):
        return fallback()
    batch = rows[0][0].shape[rows[0][1]]
    if any(x.shape[dim] != batch for x, dim in rows):
        return fallback()
    out_shape = jax.eval_shape(body, present)
    flat_out, out_tree = jax.tree.flatten(out_shape)
    if isinstance(out_layouts, str) or _is_dims(out_layouts):
        flat_out_layouts = [out_layouts] * len(flat_out)
    else:
        flat_out_layouts = [layout for layout, part in zip(out_layouts, out_shape, strict=True)
                            for _ in jax.tree.leaves(part)]

    def out_ok(layout: Any, x: Any) -> bool:
        if layout == SHARDS:
            return x.ndim >= 1
        dims = _layout_dims(layout)
        return dims is not None and dims[0] is not None and x.ndim > dims[0] and x.shape[dims[0]] == batch

    if not all(out_ok(layout, x) for layout, x in zip(flat_out_layouts, flat_out, strict=True)):
        return fallback()
    sizes = {name: int(size) for name, size in zip(mesh.axis_names, mesh.axis_sizes)}
    pairs = [*zip(in_layouts, flat_in),
             *((layout, x) for layout, x in zip(flat_out_layouts, flat_out, strict=True) if layout != SHARDS)]

    def head_dims_divide(h_axes: tuple[str, ...]) -> bool:
        head_product = 1
        for axis in h_axes:
            head_product *= sizes[axis]
        dims = ((_layout_dims(layout)[1], x) for layout, x in pairs)
        return all(dim is None or (x.ndim > dim and x.shape[dim] % head_product == 0) for dim, x in dims)

    splits = [(b, h) for b, h in _splits(mesh, batch, heads, None, head_axes) if not h or head_dims_divide(h)]
    if not splits:
        return fallback()

    def names(entry: Any) -> tuple[str, ...]:
        return () if entry is None else (entry,) if isinstance(entry, str) else tuple(entry)

    def spec(layout: Any, x: Any, b: Any, h: Any) -> P:
        if layout == SHARDS:
            return P(_entry((*names(b), *names(h))))
        batch_dim, head_dim = _layout_dims(layout)
        entries: list[Any] = [None] * x.ndim
        if batch_dim is not None:
            entries[batch_dim] = b
        if head_dim is not None and head_dim < x.ndim:
            entries[head_dim] = h
        while entries and entries[-1] is None:
            entries.pop()
        return P(*entries)

    def local(layout: Any, x: Any, b: Any, h: Any) -> jax.ShapeDtypeStruct:
        shape = list(x.shape)
        for dim, entry in enumerate(spec(layout, x, b, h)):
            for axis in names(entry):
                shape[dim] //= sizes[axis]
        return jax.ShapeDtypeStruct(tuple(shape), x.dtype)

    def accepted(b: Any, h: Any) -> bool:
        if supports is None:
            return True
        local_in = [local(layout, x, b, h) for layout, x in zip(in_layouts, flat_in, strict=True)]
        return bool(supports(**jax.tree.unflatten(in_tree, local_in), **missing))

    candidates = ((_entry(b), _entry(h)) for b, h in splits)
    chosen = next((pair for pair in candidates if accepted(*pair)), None)
    if chosen is None:
        return fallback()
    b, h = chosen

    def flat_body(*xs: Any) -> tuple[Any, ...]:
        return tuple(jax.tree.leaves(body(jax.tree.unflatten(in_tree, list(xs)))))

    mapped = jax.shard_map(
        flat_body,
        mesh=mesh,
        in_specs=tuple(spec(layout, x, b, h) for layout, x in zip(in_layouts, flat_in, strict=True)),
        out_specs=tuple(spec(layout, x, b, h) for layout, x in zip(flat_out_layouts, flat_out, strict=True)),
        check_vma=False,
        **_manual_kwargs(mesh),
    )
    return jax.tree.unflatten(out_tree, list(mapped(*flat_in)))


def rule_axes(name: str) -> tuple[str, ...]:
    """The mesh axes of the logical axis ``name`` under the rules set by :func:`use_rules` (``()``: none)."""
    return tuple(_RULES.get().get(name, ()))


def per_device_call() -> bool:
    """Whether a call traced here runs on one device: a single visible device, or the body of a ``shard_map``
    whose mesh axes are all manual (or of size 1); ``linear_cross_entropy.multi_device_reason`` is ``None``
    exactly then."""
    if jax.device_count() == 1:
        return True
    mesh = jax.sharding.get_abstract_mesh()
    return not mesh.empty and all(size == 1 or name in mesh.manual_axes for name, size in mesh.shape.items())


@contextlib.contextmanager
def per_device_step() -> Iterator[None]:
    """Mark the trace of ``train.grad_reduce = "step"``'s per-device micro-step loop
    (:func:`jla.training.grad_reduce.accumulate_per_device`): kernels that otherwise run only with one visible
    device (the fused FP8 SwiGLU passes) may run there, as on one GPU. Other fully manual ``shard_map`` bodies,
    such as the pipeline-parallel stages, keep their multi-device choice."""
    token = _PER_DEVICE_STEP.set(True)
    try:
        yield
    finally:
        _PER_DEVICE_STEP.reset(token)


def in_per_device_step() -> bool:
    """Whether a call traced here runs on one device inside :func:`per_device_step`'s micro-step loop."""
    return _PER_DEVICE_STEP.get() and per_device_call()


def mesh_can_split() -> bool:
    """Whether the context mesh has an automatic axis of size > 1, so :func:`shard_layout_call` can split a call
    traced here (without one, a call on several devices can only run unsplit)."""
    mesh = jax.sharding.get_abstract_mesh()
    return not mesh.empty and any(size > 1 and name not in mesh.manual_axes for name, size in mesh.shape.items())
