# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from deepseek-ai/DeepSeek-V3.2-Exp (MIT), inference/model.py (Indexer.forward and the MHA
# prefill branch of MLA.forward) and inference/kernel.py (fp8_index semantics).
"""Torch side of the DSA check: DeepSeek-V3.2's reference math against the JAX results of ``dsa_reference_jax.py``.

DeepSeek's inference indexer, without its FP8 quantization and Hadamard rotation (an orthogonal
rotation of q and k, so the exact scores are unchanged): ``index_score = sum_h relu(q_h . k) * w_h *
softmax_scale`` (``fp8_index``), plus the causal mask, then ``topk(min(index_topk, end_pos))``; the
MHA prefill branch adds ``index_mask`` (``-inf`` except at the top-k) and the causal mask to the
attention scores before the softmax. The indexer loss is the paper's sparse-stage KL
``sum_t KL(p_{t,S_t} || softmax(I_{t,S_t}))`` with ``p`` the attention probabilities summed over
heads and L1-normalized (the inference code has no backward; torch autograd gives every gradient).
Packed documents add the document mask to the causal mask. All math in float32 on the GPU from the
bf16 values. The reference runs twice: with its own top-k (index-set agreement with ours) and with
our indices injected (outputs and gradients of identical selections).

    python benchmarks/dsa_reference_torch.py --npz /path/dsa_ref.npz --out /path/dsa_ref.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def documents(segment_ids: torch.Tensor) -> torch.Tensor:
    """Run-numbered document ids (equal ids in separate runs are different documents)."""
    change = torch.ones_like(segment_ids, dtype=torch.int64)
    change[:, 1:] = (segment_ids[:, 1:] != segment_ids[:, :-1]).long()
    return torch.cumsum(change, dim=1)


def reference(q, k, v, q_idx, k_idx, w_idx, ct, segment_ids, top_k, indices=None):
    bsz, seqlen, n_heads, head_dim = q.shape
    group = n_heads // k.shape[2]
    doc = documents(segment_ids)
    t = torch.arange(seqlen, device=q.device)
    visible = (t[:, None] >= t[None, :])[None] & (doc[:, :, None] == doc[:, None, :])
    mask = torch.zeros(visible.shape, device=q.device).masked_fill(~visible, float("-inf"))
    # Indexer.forward (fp8_index semantics: relu(q . k) * weights, summed over heads; weights already
    # carry n_heads ** -0.5, softmax_scale = index_head_dim ** -0.5)
    softmax_scale = q_idx.shape[-1] ** -0.5
    weights = w_idx * softmax_scale
    logits = torch.einsum("bshd,btd->bsht", q_idx, k_idx).relu()
    index_score = torch.einsum("bsht,bsh->bst", logits, weights)
    index_score = index_score + mask
    if indices is None:
        topk_indices = index_score.detach().topk(min(top_k, seqlen), dim=-1)[1]
    else:  # ours, with -1 (unused) pointed at a masked key so the scatter below ignores it
        topk_indices = torch.where(indices < 0, torch.full_like(indices, seqlen - 1), indices)
        unused = indices < 0
    # MLA.forward, MHA prefill branch
    kx = k.repeat_interleave(group, dim=2)
    vx = v.repeat_interleave(group, dim=2)
    scores = torch.einsum("bshd,bthd->bsht", q, kx) * head_dim ** -0.5
    index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=q.device)
    if indices is None:
        index_mask = index_mask.scatter_(-1, topk_indices, 0)
    else:
        src = torch.where(unused, float("-inf"), 0.0).to(index_mask.dtype)
        index_mask = index_mask.scatter_reduce(-1, topk_indices, src, reduce="amax", include_self=True)
    index_mask += mask
    scores = scores + index_mask.unsqueeze(2)
    probs = scores.softmax(dim=-1)
    out = torch.einsum("bsht,bthd->bshd", probs, vx)
    # Sparse-stage indexer loss: KL(p || softmax(I)) over the selected keys, summed over queries
    selected = torch.isfinite(index_mask)
    p = probs.detach().sum(dim=2)
    p = torch.where(selected, p, 0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    log_q = torch.log_softmax(index_score.masked_fill(~selected, float("-inf")), dim=-1)
    kl = torch.where(selected & (p > 0), p * (torch.log(p.clamp_min(1e-30)) - log_q), 0.0).sum()
    loss = (out * ct).sum() + kl
    return out, loss, topk_indices, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = np.load(args.npz)
    dev = "cuda"
    names = ("q", "k", "v", "q_idx", "k_idx", "w_idx")
    top_k = int(data["top_k"])
    seg = torch.tensor(data["segment_ids"], device=dev)
    ct = torch.tensor(data["ct"], device=dev)
    ours = torch.tensor(data["indices"], device=dev).long()
    report = {"shape": {name: list(data[name].shape) for name in names}, "top_k": top_k,
              "packed": bool((data["segment_ids"] != 0).any()), "torch": torch.__version__}

    # 1. DeepSeek's own top-k: index sets per query against ours
    leaves = [torch.tensor(data[name], device=dev) for name in names]
    with torch.no_grad():
        _, _, ref_idx, selected_ref = reference(*leaves, ct, seg, top_k)
    valid = ours >= 0
    ours_sel = torch.zeros(selected_ref.shape, dtype=torch.int32, device=dev)
    ours_sel = ours_sel.scatter_add_(-1, torch.where(valid, ours, 0), valid.int()) > 0
    same_rows = (ours_sel == selected_ref).all(dim=-1)
    report["selection_rows_identical"] = float(same_rows.float().mean())
    report["selection_rows_compared"] = int(same_rows.numel())

    # 2. Our indices injected: outputs and gradients of identical selections
    leaves = [torch.tensor(data[name], device=dev, requires_grad=True) for name in names]
    out, loss, _, _ = reference(*leaves, ct, seg, top_k, indices=ours)
    loss.backward()
    want = {"out": out.detach()}
    want.update({f"d{name}": leaf.grad for name, leaf in zip(names, leaves)})
    errors = {}
    for impl in ("kernel", "xla"):
        rows = {}
        for key, ref in want.items():
            got = torch.tensor(data[f"{key}_{impl}"], device=dev)
            diff = (got - ref).abs().max().item()
            scale = ref.abs().max().item()
            rows[key] = {"max_abs": diff, "max_rel_to_max": diff / max(scale, 1e-30), "ref_max": scale}
        errors[impl] = rows
    report["errors_vs_deepseek_reference"] = errors
    print(json.dumps(report, indent=1))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
