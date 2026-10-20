# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# The float64 oracle follows the update rule of BlinkDL RWKV-LM (Apache-2.0),
# RWKV-v7/train_temp/cuda/rwkv7_clampw.cu; FLA (MIT) is called, not copied.
"""Torch side of ``rwkv7_check.py``: FLA's RWKV-7 op and layers, and a float64 oracle.

Run in a torch environment with FLA (``rwkv7_check.py --torch-python`` calls it); it reads the
inputs ``rwkv7_check.py`` wrote and writes the results next to them.

    python torch_rwkv7_check.py op --inputs case.npz --out case_torch.npz --configs oracle64,fla_exact,fla_layer
    python torch_rwkv7_check.py layer --inputs layer.npz --out layer_torch.npz

``op`` configurations, each giving ``o``, the final state and every gradient for the cotangents
``do`` and ``dht`` in the inputs:

* ``oracle64``: the RWKV-7 recurrence token by token in float64 with torch autograd (written here
  from the BlinkDL RWKV-LM CUDA kernel's update ``sa = S a; S = S diag(exp w) + sa b^T + v k^T``);
* ``fla_exact``: FLA ``chunk_rwkv7`` with its defaults (exact gates, ``chunk_size=16``);
* ``fla_layer``: ``chunk_rwkv7`` as FLA's ``RWKV7Attention`` calls it (``safe_gate=True``,
  ``lower_bound=-0.6065306597126334``, ``chunk_size=64``).

``layer`` runs FLA's ``RWKV7Attention`` (``fuse_norm`` False and True) and ``RWKV7FeedForward`` with
the weights in the inputs, forward only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.25)
_LOWER = -0.6065306597126334
_GRADS = ("r", "w", "k", "v", "a", "b", "h0")


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def oracle64(r, w, k, v, a, b, h0, cu):
    """Token-by-token recurrence in the inputs' dtype (float64 here); ``h0``/final ``[N, H, K, V]``."""
    batch, seq_len, heads, _ = r.shape
    starts = {0} if cu is None else set(cu[:-1].tolist())
    seq = 0
    state = h0[0:batch] if cu is None else h0[0:1]
    outs, finals = [], []
    # unbind: one backward node per input (slicing per token would add a full-size gradient per token)
    rs, ws, ks, vs, as_, bs = (x.unbind(1) for x in (r, w, k, v, a, b))
    for t in range(seq_len):
        if cu is not None and t in starts and t > 0:
            finals.append(state)
            seq += 1
            state = h0[seq:seq + 1]
        sa = (as_[t][..., None] * state).sum(-2, keepdim=True)  # a^T S  [B, H, 1, V]
        state = state * torch.exp(ws[t])[..., None] + bs[t][..., None] * sa + ks[t][..., None] * vs[t][..., None, :]
        outs.append((rs[t][..., None] * state).sum(-2))
    finals.append(state)
    o = torch.stack(outs, dim=1)
    final = torch.cat(finals, 0) if cu is not None else finals[0]
    return o, final


def run_op(args) -> None:
    from fla.ops.rwkv7 import chunk_rwkv7

    x = _load(args.inputs)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[str(x["dtype"])]
    cu_np = x.get("cu_seqlens")
    out: dict[str, np.ndarray] = {}
    meta = {}
    for config in args.configs.split(","):
        dt = torch.float64 if config == "oracle64" else dtype
        t = {n: torch.tensor(x[n], device="cuda", dtype=dt, requires_grad=True) for n in ("r", "w", "k", "v", "a", "b")}
        h0 = torch.tensor(x["h0"], device="cuda", dtype=torch.float64 if config == "oracle64" else torch.float32,
                          requires_grad=True)
        cu = None if cu_np is None else torch.tensor(cu_np, device="cuda", dtype=torch.long)
        try:
            if config == "oracle64":
                o, final = oracle64(t["r"], t["w"], t["k"], t["v"], t["a"], t["b"], h0, cu_np)
            else:
                extra = dict(safe_gate=True, lower_bound=_LOWER, chunk_size=64) if config == "fla_layer" else {}
                o, final = chunk_rwkv7(r=t["r"], w=t["w"], k=t["k"], v=t["v"], a=t["a"], b=t["b"], scale=1.0,
                                       initial_state=h0, output_final_state=True, cu_seqlens=cu, **extra)
            do = torch.tensor(x["do"], device="cuda", dtype=o.dtype)
            dht = torch.tensor(x["dht"], device="cuda", dtype=final.dtype)
            grads = torch.autograd.grad([o, final], [t[n] for n in _GRADS[:-1]] + [h0], grad_outputs=[do, dht],
                                        allow_unused=True)
        except Exception as err:  # noqa: BLE001  (record the failure, keep the other configurations)
            meta[config] = f"{type(err).__name__}: {str(err)[:300]}"
            continue
        save_dt = np.float64 if config == "oracle64" else np.float32
        out[f"{config}/o"] = o.detach().to(torch.float64).cpu().numpy().astype(save_dt)
        out[f"{config}/final"] = final.detach().to(torch.float64).cpu().numpy().astype(save_dt)
        for name, g in zip(_GRADS, grads):
            if g is not None:
                out[f"{config}/d{name}"] = g.detach().to(torch.float64).cpu().numpy().astype(save_dt)
        meta[config] = "ok"
    np.savez(args.out, **out)
    import fla
    print(json.dumps(dict(configs=meta, fla=getattr(fla, "__version__", "?"), torch=torch.__version__)))


def run_layer(args) -> None:
    from fla.layers.rwkv7 import RWKV7Attention
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward

    x = _load(args.inputs)
    cfg = json.loads(str(x["config"]))
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["dtype"]]
    dev = "cuda"
    hidden = torch.tensor(x["x"], device=dev, dtype=dtype)
    cu = None if "cu_seqlens" not in x else torch.tensor(x["cu_seqlens"], device=dev, dtype=torch.long)
    v_first = None if "v_first" not in x else torch.tensor(x["v_first"], device=dev, dtype=dtype)
    out = {}
    for fuse_norm in (False, True):
        att = RWKV7Attention(
            mode="chunk", hidden_size=cfg["d_model"], head_dim=cfg["head_dim"], layer_idx=cfg["layer_idx"],
            num_hidden_layers=cfg["n_layers"], decay_low_rank_dim=cfg["lora"][0], a_low_rank_dim=cfg["lora"][1],
            v_low_rank_dim=cfg["lora"][2], gate_low_rank_dim=cfg["lora"][3], norm_eps=cfg["norm_eps"],
            fuse_norm=fuse_norm,
        ).to(dev)
        state = att.state_dict()
        for name in list(state):
            key = "att." + name.replace("g_norm.", "norm.")
            if key not in x:
                raise KeyError(f"no weight for {name}")
            state[name] = torch.tensor(x[key]).reshape(state[name].shape)
        att.load_state_dict(state)
        att = att.to(dtype).eval()
        with torch.no_grad():
            o, _, _, vf = att(hidden, v_first=v_first, cu_seqlens=cu)
        tag = "fused" if fuse_norm else "plain"
        out[f"att_{tag}"] = o.float().cpu().numpy()
        out[f"v_first_{tag}"] = vf.float().cpu().numpy()
    ffn = RWKV7FeedForward(hidden_size=cfg["d_model"], intermediate_size=cfg["d_ff"], hidden_act="sqrelu",
                           layer_idx=cfg["layer_idx"], num_hidden_layers=cfg["n_layers"]).to(dev)
    state = ffn.state_dict()
    for name in list(state):
        state[name] = torch.tensor(x["ffn." + name]).reshape(state[name].shape)
    ffn.load_state_dict(state)
    ffn = ffn.to(dtype).eval()
    with torch.no_grad():
        f, _ = ffn(hidden, cu_seqlens=cu)
    out["ffn"] = f.float().cpu().numpy()
    np.savez(args.out, **out)
    import fla
    print(json.dumps(dict(layer="ok", fla=getattr(fla, "__version__", "?"), torch=torch.__version__)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("op", "layer"))
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default="oracle64,fla_exact,fla_layer")
    args = parser.parse_args()
    run_op(args) if args.what == "op" else run_layer(args)


if __name__ == "__main__":
    main()
