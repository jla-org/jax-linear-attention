# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The cheaper pieces of the delta rule's exact sequence split.

* the forward map pass with two row blocks per CTA (``scalar_gate_docs.PAIRED_MAPS``) against the
  single-block kernel, and without an initial state (the first piece's ``M`` is not computed);
* the identity blocks of pieces with a document start hold ``M = 0`` without walking;
* the scans that start from a zero state or gradient take the local terms directly;
* ``kkt_fwd``'s chunks per CTA (any count gives the same inverse; the default policy);
* end to end: no initial state, with and without a final state, dense and packed documents,
  ``K = V`` in {64, 128}, grouped key heads, against chunked autodiff (values, final state and every
  gradient) and against the previous full walks;
* the opt-in warm-up split (``cp_warmup``, FlashQLA's approximation): its windows, its error, and the exact
  split when no window reaches the threshold;
* (20 Oct) the piece scan kernel (``scalar_gate_scan``) against the XLA scan, with and without zero maps; the
  skipped forward maps of pieces followed by a document start; 8 pieces end to end (dense, documents aligned
  with the pieces and not, no state or both states, grouped key heads, extreme gates, an odd length) against
  chunked autodiff, bit for bit against the full walks, and against the XLA scan; the default policy's 8 pieces
  with two rows and documents (fast and slow gates).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import gated_delta_rule_bwd as gb
from jla.ops.mosaic_gpu import gated_delta_rule_kernels as gk
from jla.ops.mosaic_gpu import scalar_gate as sg
from jla.ops.mosaic_gpu import scalar_gate_docs as sgd

gpu = pytest.mark.skipif(not cm.is_hopper(), reason="needs an H100 (compute capability 9.0)")
F32, BF16 = jnp.float32, jnp.bfloat16
HIGHEST = lax.Precision.HIGHEST


def _inputs(seed, B, T, H, K=128, V=128, Hk=None):
    ks = jax.random.split(jax.random.key(seed), 8)
    Hk = H if Hk is None else Hk
    return dict(
        q=jax.random.normal(ks[0], (B, T, Hk, K)).astype(BF16),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T, Hk, K))).astype(BF16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(BF16),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0),
        beta=jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H))),
        do=jax.random.normal(ks[6], (B, T, H, V)),
        dht=jax.random.normal(ks[7], (B, H, K, V)),
    )


def _segment_ids(B, T, pattern):
    """Document layouts: none, unaligned starts (some pieces without one), or many short documents."""
    if pattern == "none":
        return None
    rng = np.random.default_rng(1)
    rows = np.zeros((B, T), np.int32)
    for b in range(B):
        if pattern == "unaligned":
            starts = np.sort(np.asarray([[100, 300, 333, 650, 900], [37, 700]][b % 2]) * T // 1024)
        else:  # "tiny": mean length 5
            starts = np.sort(rng.choice(np.arange(1, T), size=T // 5, replace=False))
        rows[b] = np.searchsorted(starts, np.arange(T), side="right")
    return jnp.asarray(rows)


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def test_paired_map_policy(monkeypatch):
    monkeypatch.setattr(sgd, "num_sms", lambda: 132)
    assert sgd._use_pairs(160, False)  # (1, 16384, 16) at cp = 4: two waves of single blocks, one of pairs
    assert not sgd._use_pairs(80, False)  # one wave either way
    assert not sgd._use_pairs(300, False)  # three waves against two of pairs at 1.64x
    assert not sgd._use_pairs(160, True)  # packed documents: the identity blocks mostly walk nothing


def test_kkt_chunks_per_cta_policy(monkeypatch):
    monkeypatch.setattr(gk, "num_sms", lambda: 132)
    assert gk.kkt_chunks_per_cta(1, 8, 64) == 2  # 64 chunks x 8 heads: 256 CTAs of 2 chunks
    assert gk.kkt_chunks_per_cta(8, 32, 32) == 8
    assert gk.kkt_chunks_per_cta(1, 16, 256) == 8
    assert gk.kkt_chunks_per_cta(4, 16, 64) == 8
    assert gk.kkt_chunks_per_cta(1, 1, 3) == 1
    assert gk.kkt_chunks_per_cta(1, 4, 100) == 2  # 8 does not divide 100; 4 per CTA leaves 100 CTAs, 2 leave 200


@gpu
@pytest.mark.parametrize("pattern", ["none", "tiny"])
def test_kkt_fwd_any_chunks_per_cta(pattern):
    B, T, H = 1, 1024, 3
    x = _inputs(30, B, T, H)
    bt = cm.head_major(x["beta"], 0)
    seg = None if pattern == "none" else cm.segment_vectors(_segment_ids(B, T, pattern), 0)
    outs = [np.asarray(jax.jit(lambda k, b, n=n: gk.kkt_fwd(k, b, seg=seg, chunks_per_cta=n))(x["k"], bt))
            for n in (8, 4, 2, 1)]
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


@gpu
@pytest.mark.parametrize("pattern", ["none", "unaligned", "tiny"])
def test_paired_map_pass_matches_single_blocks(pattern, monkeypatch):
    """Two row blocks per CTA give the single-block kernel's maps; without ``need_first`` the first
    piece's ``M`` is zero and everything else is unchanged; pieces with a document start hold ``M = 0``."""
    B, T, H, cp, K = 2, 1024, 2, 4, 128
    x = _inputs(31, B, T, H)
    g2 = cm.chunk_log2_decay(x["g"], B, T, H, 0)
    bt = cm.head_major(x["beta"], 0)
    segment_ids = _segment_ids(B, T, pattern)
    seg = None if segment_ids is None else cm.segment_vectors(segment_ids, 0)

    def maps(paired, need_first, skip_fresh=False):
        monkeypatch.setattr(sgd, "_use_pairs", lambda blocks, docs: paired)
        f = jax.jit(lambda k, v, g2, bt: sgd.map_fwd_wy(k, v, gk.kkt_fwd(k, bt, seg=seg), g2, bt, seg=seg, cp=cp,
                                                        need_first=need_first, skip_fresh=skip_fresh))
        return np.asarray(f(x["k"], x["v"], g2, bt)).reshape(B, cp, H, 2 * K, K)

    @jax.jit
    def full_walk(k, v, g2, bt):
        _, end = gb.state_fwd_wy(k, v, gk.kkt_fwd(k, bt, seg=seg), g2, bt, None, store_states=False,
                                 out_final=True, aug=True, seg=seg, cp=cp)
        return end

    full = np.asarray(full_walk(x["k"], x["v"], g2, bt)).reshape(B, cp, H, 2 * K, K)
    paired, single = maps(True, True), maps(False, True)
    np.testing.assert_array_equal(paired, single)  # the same operations per row block: bit for bit
    np.testing.assert_allclose(paired, full, rtol=0, atol=1e-6)
    no_first = maps(True, False)
    np.testing.assert_array_equal(no_first, maps(False, False))
    np.testing.assert_array_equal(no_first[:, 0, :, :K], 0)
    np.testing.assert_allclose(no_first[:, 0, :, K:], full[:, 0, :, K:], rtol=0, atol=1e-6)
    np.testing.assert_allclose(no_first[:, 1:], full[:, 1:], rtol=0, atol=1e-6)
    if seg is not None:  # a piece with a document start passes nothing on: M = 0 exactly
        killed = np.asarray(seg.cin[:, cm.CHUNK - 1::cm.CHUNK] == 0).reshape(B, cp, -1).any(-1)
        assert killed.any()
        np.testing.assert_array_equal(paired[killed][:, :, :K], 0)
    # The default skips the maps of pieces whose successor starts a document at its first token
    # (they hold [0 | 0]); every other piece is unchanged, bit for bit.
    fresh = np.asarray(sgd.fresh_next(seg, B, T, cp))
    skipped = maps(True, True, skip_fresh=True)
    np.testing.assert_array_equal(skipped, maps(False, True, skip_fresh=True))
    np.testing.assert_array_equal(skipped[~fresh], paired[~fresh])
    np.testing.assert_array_equal(skipped[fresh], 0)


def _value_and_grads(f, x, names):
    def loss(p):
        o, s = f(p)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), (o, s)

    (_, (o, s)), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))({n: x[n] for n in names})
    return o, s, grads


@gpu
@pytest.mark.parametrize("final", [True, False], ids=["final_state", "no_final_state"])
@pytest.mark.parametrize("pattern", ["none", "unaligned", "tiny"])
@pytest.mark.parametrize("dims", [(128, 128, 2), (64, 64, 2), (128, 128, 1)], ids=["k128", "k64", "k128_grouped"])
def test_split_without_initial_state(dims, pattern, final, monkeypatch):
    """No initial state at cp = 4 (and 300 tokens at cp = 2, padded): values, final state and every gradient
    against chunked autodiff and the previous full walks. At K = V = 128 the paired map kernel is forced."""
    K, V, Hk = dims
    if K == 128:
        monkeypatch.setattr(sgd, "_use_pairs", lambda blocks, docs: True)
    for B, T, H, cp in ((2, 1024, 2, 4), (1, 300, 2, 2)):
        x = _inputs(32, B, T, H, K, V, Hk=Hk)
        segment_ids = _segment_ids(B, T, pattern)
        names = ("q", "k", "v", "g", "beta")
        zero = jnp.zeros((B, H, K, V), F32)

        def run(walk):
            def call(p):
                o, s = sg.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None, initial_state=None,
                              output_final_state=final, segment_ids=segment_ids, cp=cp, walk=walk)
                return o, (s if final else zero)
            return call

        def chunked(p):
            q, k = (jnp.repeat(p[n], H // Hk, axis=2).astype(F32) for n in ("q", "k"))
            o, s = ops.linear_attention("gated_deltanet", q, k, p["v"].astype(F32), impl="chunked",
                                        g=p["g"], beta=p["beta"], output_final_state=True,
                                        segment_ids=segment_ids, precision=HIGHEST)
            return o, (s if final else zero)

        o, s, got = _value_and_grads(run(True), x, names)
        o_full, s_full, full = _value_and_grads(run(False), x, names)
        o_ref, s_ref, want = _value_and_grads(chunked, x, names)
        _assert_close(o, o_ref, 2e-2, "o")
        _assert_close(o, o_full, 0.0, "o vs full walk")  # bit for bit
        if final:
            _assert_close(s, s_ref, 2e-2, "final state")
            _assert_close(s, s_full, 0.0, "final state vs full walk")
        for name in names:
            _assert_close(got[name], want[name], 2e-2, f"d{name}")
            _assert_close(got[name], full[name], 0.0, f"d{name} vs full walk")


def test_warmup_windows():
    """Chunks from the piece end (forward) or start (backward) until the decay first falls below the threshold."""
    B, H, cp, ncs = 1, 2, 2, 4
    per_chunk = np.array([[-1.0, -3.0, -0.5, -8.0, -0.1, -0.1, -0.1, -0.1],  # head 0: piece 0, then piece 1
                          [-20.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0]], np.float32)  # head 1
    g2 = np.zeros((B, H, cp * ncs * cm.CHUNK), np.float32)
    g2[0, :, cm.CHUNK - 1::cm.CHUNK] = per_chunk
    fwd = np.asarray(sgd.warmup_windows(jnp.asarray(g2), cp, -8.5, trailing=True)).reshape(cp, H)
    bwd = np.asarray(sgd.warmup_windows(jnp.asarray(g2), cp, -8.5, trailing=False)).reshape(cp, H)
    np.testing.assert_array_equal(fwd, [[3, 4], [0, 0]])  # head 0, piece 0 from its end: -8, -8.5 (not below), -11.5
    np.testing.assert_array_equal(bwd, [[4, 1], [0, 0]])


def _gdn_inputs(seed, B, T, H, slow):
    x = _inputs(seed, B, T, H)
    if slow:  # per-token decay 0.999: no window reaches e^-10 within a piece, so the split stays exact
        x["g"] = jnp.full((B, T, H), np.log(0.999), F32)
    else:  # the benchmark's gates: logsigmoid(N(0, 1)), about -0.8 per token
        x["g"] = jax.nn.log_sigmoid(jax.random.normal(jax.random.key(seed + 1), (B, T, H)))
    return x


@gpu
@pytest.mark.parametrize("paired", [False, True], ids=["single", "paired"])
@pytest.mark.parametrize("pattern", ["none", "unaligned"])
@pytest.mark.parametrize("slow", [False, True], ids=["fast_gates", "slow_gates"])
def test_cp_warmup(slow, pattern, paired, monkeypatch):
    """The opt-in warm-up split: within bf16 noise of chunked autodiff with fast gates, and the exact split
    itself (same values) when no window reaches the threshold; with an initial and a final state, with the
    single-block and (forced) the paired forward map kernel."""
    if paired:
        monkeypatch.setattr(sgd, "_use_pairs", lambda blocks, docs: True)
    B, T, H, cp = 2, 1024, 2, 4
    x = _gdn_inputs(33, B, T, H, slow)
    x["h0"] = 0.1 * jax.random.normal(jax.random.key(5), (B, H, 128, 128))
    segment_ids = _segment_ids(B, T, pattern)
    names = ("q", "k", "v", "g", "beta", "h0")

    def run(warmup):
        def call(p):
            return sg.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None, initial_state=p["h0"],
                          output_final_state=True, segment_ids=segment_ids, cp=cp, cp_warmup=warmup)
        return call

    def chunked(p):
        return ops.linear_attention("gated_deltanet", *(p[n].astype(F32) for n in ("q", "k", "v")), impl="chunked",
                                    g=p["g"], beta=p["beta"], initial_state=p["h0"], output_final_state=True,
                                    segment_ids=segment_ids, precision=HIGHEST)

    o, s, got = _value_and_grads(run(-10.0), x, names)
    o_exact, s_exact, exact = _value_and_grads(run(None), x, names)
    o_ref, s_ref, want = _value_and_grads(chunked, x, names)
    if slow:
        _assert_close(o, o_exact, 0.0, "o vs exact split")
        _assert_close(s, s_exact, 0.0, "final state vs exact split")
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s, s_ref, 2e-2, "final state")
    for name in names:
        if slow:
            _assert_close(got[name], exact[name], 0.0, f"d{name} vs exact split")
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


def test_cp_warmup_guards():
    x = _inputs(34, 1, 128, 1)
    with pytest.raises(ValueError, match="cp_warmup"):
        sg.run(x["q"], x["k"], x["v"], x["g"], x["beta"], delta=True, scale=None, initial_state=None,
               output_final_state=False, cp=2, wy=False, cp_warmup=-10.0)
    for bad in (0.0, 10.0, float("nan")):  # a threshold of 0 or above would silently drop the carried state
        with pytest.raises(ValueError, match="cp_warmup must be a negative"):
            sg.run(x["q"], x["k"], x["v"], x["g"], x["beta"], delta=True, scale=None, initial_state=None,
                   output_final_state=False, cp=2, cp_warmup=bad)


# ------------------------------------------------------------------ the piece scan kernel and eight pieces (20 Oct)

from jla.ops.mosaic_gpu import scalar_gate_scan as sgs  # noqa: E402


def _scan_inputs(seed, B, cp, H, K, V):
    """``[M^T | E^T]`` of every piece (one array, as the forward map pass writes it), an initial state and zero-map
    flags; a flagged piece's ``M`` is exactly zero, as the walk leaves it."""
    ks = jax.random.split(jax.random.key(seed), 4)
    end = jax.random.normal(ks[0], (B, cp, H, K + V, K), F32)
    end = end.at[:, :, :, :K].multiply(0.9 / np.sqrt(K))  # maps of norm about 1
    zero = (jax.random.uniform(ks[1], (B, cp, H)) < 0.4).astype(jnp.int32)
    end = end.at[:, :, :, :K].set(jnp.where(zero[..., None, None] != 0, 0.0, end[:, :, :, :K]))
    init = jax.random.normal(ks[2], (B, H, V, K), F32)
    return end, zero, init


def _scan_cases(interpret):
    cases = ((2, 3, 2, 64, 64), (1, 4, 1, 128, 128)) if interpret else (
        (1, 2, 8, 64, 64), (2, 8, 4, 128, 128), (1, 16, 8, 128, 128), (1, 8, 3, 128, 64))
    for B, cp, H, K, V in cases:
        for reverse in (False, True):
            for with_init in (False, True):
                yield B, cp, H, K, V, reverse, with_init


def _check_scan(interpret):
    for B, cp, H, K, V, reverse, with_init in _scan_cases(interpret):
        end, zero, init = _scan_inputs(B * 100 + cp, B, cp, H, K, V)
        init = init if with_init else None
        row0 = 0 if reverse else K  # backward: the local terms are their own array
        local = end[:, :, :, K:] if reverse else end
        got = sgs.piece_scan(local, end, init, zero, value_dim=V, local_row0=row0, reverse=reverse,
                             interpret=interpret)
        # The XLA scan with every product computed (the flagged maps are zero): the skip changes nothing.
        want = sgs.piece_scan_xla(local, end, init, jnp.zeros_like(zero), value_dim=V, local_row0=row0,
                                  reverse=reverse)
        _assert_close(got, want, 1e-5, f"scan {B, cp, H, K, V} reverse={reverse} init={with_init}")
        skipped = sgs.piece_scan_xla(local, end, init, zero, value_dim=V, local_row0=row0, reverse=reverse)
        _assert_close(skipped, want, 0.0, "the XLA scan's skip")


@gpu
def test_piece_scan_kernel_matches_xla_scan():
    """The Pallas Triton scan against the XLA scan: forward and reverse, with and without an initial state
    (gradient), 2-16 pieces, ``K, V`` in {64, 128}, zero maps skipped."""
    _check_scan(False)


def test_piece_scan_kernel_in_interpreter():
    _check_scan(True)


def _segment_ids_aligned(T, cp):
    """Documents that start exactly at some piece boundaries (their predecessors' maps are skipped), one
    piece-internal start, and one piece with no start at all."""
    piece = T // cp
    starts = sorted({piece, 2 * piece + 37, 3 * piece, 5 * piece, 5 * piece + 3, 7 * piece - 64} & set(range(1, T)))
    return jnp.asarray(np.searchsorted(np.asarray(starts), np.arange(T), side="right")[None].astype(np.int32))


def test_fresh_next_pieces():
    T, cp = 1024, 8
    seg = cm.segment_vectors(_segment_ids_aligned(T, cp), 0)
    fresh = np.asarray(sgd.fresh_next(seg, 1, T, cp))[0]
    assert fresh.tolist() == [True, False, True, False, True, False, False, False]
    table = sgd.fwd_walk_table(seg, jnp.zeros((1, 2, T), F32), cp, need_last=True, need_first=True)
    zero = np.asarray(sgd.zero_maps(table))[:, 0]
    counts = np.asarray(table)[:, 0, 1]
    assert (counts[fresh] == 0).all() and (zero[fresh] == 1).all()
    assert zero.tolist() == [1, 1, 1, 1, 1, 1, 1, 0]  # every piece but the last has a start or a fresh successor


@gpu
@pytest.mark.parametrize("gates", ["normal", "extreme"])
@pytest.mark.parametrize("pattern", ["none", "unaligned", "aligned"])
@pytest.mark.parametrize("state", ["none", "initial_final"])
@pytest.mark.parametrize("dims", [(128, 128, 2), (64, 64, 1)], ids=["k128", "k64_grouped"])
def test_eight_pieces(dims, state, pattern, gates):
    """8 pieces (the default at ``B * H * V / 64 <= 16`` since 20 Oct) with the kernel scan: values, final state
    and every gradient (``dh0`` too) against chunked autodiff, bit for bit against the full walks, and against
    the previous XLA scan."""
    _check_eight_pieces(dims, state, pattern, gates, 2048)


@gpu
def test_eight_pieces_odd_length():
    """An odd length (1900 tokens, padded to 8 pieces of 256) with documents and both states."""
    _check_eight_pieces((128, 128, 2), "initial_final", "unaligned", "normal", 1900)


def _check_eight_pieces(dims, state, pattern, gates, T):
    K, V, Hk = dims
    B, H, cp = 1, 2, 8
    x = _inputs(40, B, T, H, K, V, Hk=Hk)
    if gates == "extreme":  # one head without decay, one that forgets everything within a few tokens
        x["g"] = jnp.stack([jnp.zeros((B, T)), jnp.full((B, T), -30.0)], axis=-1).astype(F32)
    segment_ids = {"none": None, "unaligned": _segment_ids(B, T, "unaligned"),
                   "aligned": _segment_ids_aligned(T, cp)}[pattern]
    x["h0"] = 0.5 * jax.random.normal(jax.random.key(41), (B, H, K, V), F32)
    with_state = state == "initial_final"
    names = ("q", "k", "v", "g", "beta") + (("h0",) if with_state else ())
    zero = jnp.zeros((B, H, K, V), F32)

    def run(walk, scan_kernel=True):
        def call(p):
            o, s = sg.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None,
                          initial_state=p["h0"] if with_state else None, output_final_state=with_state,
                          segment_ids=segment_ids, cp=cp, walk=walk, piece_scan_kernel=scan_kernel)
            return o, (s if with_state else zero)
        return call

    def chunked(p):
        q, k = (jnp.repeat(p[n], H // Hk, axis=2).astype(F32) for n in ("q", "k"))
        o, s = ops.linear_attention("gated_deltanet", q, k, p["v"].astype(F32), impl="chunked", g=p["g"],
                                    beta=p["beta"], initial_state=p["h0"] if with_state else None,
                                    output_final_state=True, segment_ids=segment_ids, precision=HIGHEST)
        return o, (s if with_state else zero)

    o, s, got = _value_and_grads(run(True), x, names)
    o_full, s_full, full = _value_and_grads(run(False), x, names)
    o_ref, s_ref, want = _value_and_grads(chunked, x, names)
    o_xla, s_xla, xla = _value_and_grads(run(True, scan_kernel=False), x, names)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(o, o_full, 0.0, "o vs full walk")
    _assert_close(o, o_xla, 1e-2, "o vs the XLA scan")
    if with_state:
        _assert_close(s, s_ref, 2e-2, "final state")
        _assert_close(s, s_full, 0.0, "final state vs full walk")
        _assert_close(s, s_xla, 1e-2, "final state vs the XLA scan")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")
        _assert_close(got[name], full[name], 0.0, f"d{name} vs full walk")
        _assert_close(got[name], xla[name], 1e-2, f"d{name} vs the XLA scan")


@gpu
@pytest.mark.parametrize("gates", ["normal", "slow"])
def test_default_eight_pieces_two_rows_with_documents(gates):
    """The default policy's 8 pieces (``cp`` not given) with ``B = 2``: documents starting at piece boundaries
    (fresh successors, skipped maps), inside pieces and at different places per row, both states. Values, final
    state and every gradient are bit for bit those of the full walks, and close to the XLA piece scan and the
    unsplit call (``slow``: one head without decay, so the incoming states matter)."""
    B, T, H, K, V = 2, 16384, 4, 128, 128
    assert sg.pieces(B, H, V, T, delta=True) == 8
    x = _inputs(50, B, T, H, K, V)
    if gates == "slow":  # head 0 without decay, the others decaying slowly
        slow = -jax.nn.softplus(jax.random.normal(jax.random.key(52), (B, T, H - 1)) - 6.0)
        x["g"] = jnp.concatenate([jnp.zeros((B, T, 1)), slow], axis=-1).astype(F32)
    x["h0"] = 0.5 * jax.random.normal(jax.random.key(51), (B, H, K, V), F32)
    piece = T // 8
    starts = np.zeros((B, T), np.int32)
    starts[0, [piece, 3 * piece, 3 * piece + 777, 6 * piece]] = 1
    starts[1, [100, 2 * piece + 64, 5 * piece, 7 * piece + 1]] = 1
    segment_ids = jnp.asarray(np.cumsum(starts, axis=1).astype(np.int32))
    names = ("q", "k", "v", "g", "beta", "h0")

    def run(**options):
        def call(p):
            return sg.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None, initial_state=p["h0"],
                          output_final_state=True, segment_ids=segment_ids, **options)
        return call

    o, s, got = _value_and_grads(run(), x, names)
    o_full, s_full, full = _value_and_grads(run(walk=False), x, names)
    o_xla, s_xla, xla = _value_and_grads(run(piece_scan_kernel=False, skip_fresh_maps=False), x, names)
    o_one, s_one, one = _value_and_grads(run(cp=1), x, names)
    for name, a, b in [("o", o, o_full), ("final state", s, s_full)] + [(f"d{n}", got[n], full[n]) for n in names]:
        _assert_close(a, b, 0.0, f"{name} vs full walk")
    for name, a, b in [("o", o, o_xla), ("final state", s, s_xla)] + [(f"d{n}", got[n], xla[n]) for n in names]:
        _assert_close(a, b, 1e-2, f"{name} vs the XLA scan")
    for name, a, b in [("o", o, o_one), ("final state", s, s_one)] + [(f"d{n}", got[n], one[n]) for n in names]:
        _assert_close(a, b, 3e-2, f"{name} vs no split")
