# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Linear-attention and state-space ops with selectable implementations.

    o, final_state = linear_attention("gated_deltanet", q, k, v, g=g, beta=beta)

Kinds and their gate inputs (all ``[B, T, H]`` unless noted, FLA layout):

* ``simple_gla`` (``g``): one log-decay per head; Mamba-2's SSD core with
  ``q=C, k=B, v=dt*x, g=dt*A, scale=1``;
* ``gla`` (``g [B, T, H, K]``): a log-decay per key channel;
* ``deltanet`` (``beta``); ``gated_deltanet`` (``g, beta``); ``kda``
  (``g [B, T, H, K], beta``);
* ``gdn2`` (``g [B, T, H, K], b [B, T, H, K], w [B, T, H, V]``): Gated DeltaNet-2;
* ``mamba3`` (``g, beta, gamma``): Mamba-3 SISO trapezoidal recurrence; its
  state is a :class:`~jla.ops.reference.Mamba3State`.
* ``mamba3_mimo`` (``g, beta, gamma``): Mamba-3 MIMO of rank ``R`` (``q, k [B, T, H, R, K]``,
  ``v [B, T, H, R, V]``); ``R = 1`` is ``mamba3``.

Implementations (``impl``):

* ``"reference"``: token-by-token ``lax.scan``; the oracle;
* ``"chunked"``: chunk-parallel plain JAX (``jla.ops.chunked``);
* ``"mosaic_gpu"``: Pallas Mosaic GPU kernels, once registered for a kind;
* ``"auto"`` (default): the first of ``mosaic_gpu``, ``chunked``, ``reference``
  that is registered as auto-eligible and whose ``supports`` predicate accepts
  the call.

Kernels register themselves with :func:`register`; they join ``"auto"`` only
once they are qualified (``auto=True``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from jla.ops import chunked, reference

__all__ = ["KINDS", "available", "chunked", "linear_attention", "reference", "register"]

Array = jax.Array

# kind -> (function name in reference/chunked, gate inputs in positional order)
KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "simple_gla": ("simple_gla", ("g",)),
    "gla": ("gla", ("g",)),
    "deltanet": ("delta_rule", ("beta",)),
    "gated_deltanet": ("gated_delta_rule", ("g", "beta")),
    "kda": ("kda", ("g", "beta")),
    "gdn2": ("gdn2", ("g", "b", "w")),
    "mamba3": ("mamba3", ("g", "beta", "gamma")),
    "mlstm": ("mlstm", ("g", "i")),
    "log_linear_simple_gla": ("log_linear_simple_gla", ("g", "level_scales")),
    "mamba1": ("mamba1", ("dt", "a")),  # q = C, k = B, v = u; a = A [D, N]
    "gated_deltaproduct": ("gated_delta_product", ("g", "beta")),
    "mamba3_mimo": ("mamba3_mimo", ("g", "beta", "gamma")),
}
_GATE_NAMES = frozenset(name for _, gates in KINDS.values() for name in gates)

_ORDER = ("mosaic_gpu", "chunked", "reference")
_Supports = Callable[..., bool]
_REGISTRY: dict[str, dict[str, tuple[Callable[..., Any], _Supports, bool]]] = {name: {} for name in _ORDER}


def register(
    impl: str,
    kind: str,
    fn: Callable[..., Any],
    supports: _Supports | None = None,
    *,
    auto: bool = False,
) -> None:
    """Register ``fn`` as implementation ``impl`` of ``kind``.

    ``fn`` takes ``(q, k, v, *gates, **options)`` exactly like the reference
    function of the same kind, with the gates in ``KINDS[kind]`` order.
    ``supports(q, k, v, *gates, **options)`` returns False for calls the
    implementation cannot handle; ``"auto"`` then falls through to the next one.

    ``auto=True`` makes the implementation eligible for ``impl="auto"``. Kernels
    pass it only once they are tested on GPU for every call ``supports`` accepts
    and are faster than the chunked path; until then they run only when
    requested by name (``impl="mosaic_gpu"``).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown linear-attention kind {kind!r}")
    _REGISTRY.setdefault(impl, {})[kind] = (fn, supports or (lambda *args, **options: True), auto)


_kernels_loaded = False


def _load_kernels() -> None:
    """Import the optional Mosaic GPU kernels once; each module registers itself.

    A missing ``jla.ops.mosaic_gpu`` package is fine (the kernels are
    optional); any other import error is a bug and is raised.
    """
    global _kernels_loaded
    if _kernels_loaded:
        return
    _kernels_loaded = True
    try:
        import jla.ops.mosaic_gpu  # noqa: F401
    except ModuleNotFoundError as err:
        if err.name != "jla.ops.mosaic_gpu":
            raise


def available(kind: str) -> tuple[str, ...]:
    """Registered implementations of ``kind``, fastest first."""
    _load_kernels()
    return tuple(impl for impl in _ORDER if kind in _REGISTRY.get(impl, {}))


def _no_packed(*args: Any, cu_seqlens: Array | None = None, **options: Any) -> bool:
    return cu_seqlens is None


for _kind, (_name, _) in KINDS.items():
    register("reference", _kind, getattr(reference, _name), auto=True)
    register("chunked", _kind, getattr(chunked, _name), _no_packed, auto=True)


def linear_attention(
    kind: str,
    q: Array,
    k: Array,
    v: Array,
    *,
    impl: str = "auto",
    **inputs: Any,
) -> tuple[Array, Any]:
    """Run a linear-attention op; returns ``(o, final_state)``.

    ``inputs`` holds the kind's gates by name (``g``, ``beta``, ``b``, ``w``,
    ``gamma``) plus options passed through: ``scale``, ``initial_state``,
    ``output_final_state``, ``segment_ids``, ``cu_seqlens`` (reference only),
    ``use_qk_l2norm`` (delta-rule kinds), and implementation-specific ones such
    as ``chunk_size``, which the reference ignores.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown linear-attention kind {kind!r}; expected one of {sorted(KINDS)}")
    _, gate_names = KINDS[kind]
    missing = [name for name in gate_names if inputs.get(name) is None]
    extra = sorted(name for name in _GATE_NAMES - set(gate_names) if inputs.get(name) is not None)
    if missing or extra:
        raise ValueError(f"{kind} takes gates {gate_names}; missing {missing}, unexpected {extra}")
    gates = [inputs.pop(name) for name in gate_names]
    options = {key: value for key, value in inputs.items() if key not in _GATE_NAMES}
    if impl in ("auto", "mosaic_gpu"):
        _load_kernels()
    candidates = _ORDER if impl == "auto" else (impl,)
    for name in candidates:
        entry = _REGISTRY.get(name, {}).get(kind)
        if entry is None:
            continue
        fn, supports, auto_eligible = entry
        call_options = dict(options)
        if name == "reference":
            for key in ("chunk_size", "sub_chunk", "precision"):
                call_options.pop(key, None)
        if impl == "auto" and not (auto_eligible and supports(q, k, v, *gates, **call_options)):
            continue
        if name == "mosaic_gpu":  # custom calls XLA cannot partition: run them per shard of the context mesh
            from jla.distributed.kernel_sharding import shard_kernel_call

            return shard_kernel_call(fn, q, k, v, gates, call_options, supports=supports, kind=kind)
        return fn(q, k, v, *gates, **call_options)
    raise ValueError(f"no {impl!r} implementation of {kind!r} accepts this call (have {available(kind)})")


# RWKV-6 (GLA plus a current-token bonus ``u [H, K]``) lives in its own module.
from jla.ops import rwkv6  # noqa: E402

KINDS["rwkv6"] = ("rwkv6", ("g", "u"))
_GATE_NAMES = frozenset(name for _, gates in KINDS.values() for name in gates)
register("reference", "rwkv6", rwkv6.reference, auto=True)
register("chunked", "rwkv6", rwkv6.chunked, _no_packed, auto=True)


# RWKV-7 (generalised delta rule, DPLR): ``w`` (log decay), ``a``, ``b`` all ``[B, T, H, K]``;
# the reference and chunked ops live in jla.ops.rwkv7.
from jla.ops import rwkv7 as _rwkv7  # noqa: E402

KINDS["rwkv7"] = ("rwkv7", ("w", "a", "b"))
_GATE_NAMES = frozenset(name for _, gates in KINDS.values() for name in gates)
register("reference", "rwkv7", _rwkv7.rwkv7, auto=True)
register("chunked", "rwkv7", _rwkv7.rwkv7_chunked, _no_packed, auto=True)
