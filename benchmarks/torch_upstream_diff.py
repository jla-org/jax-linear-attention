# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Upstream side of ``diff_upstream.py``: one case through the latest upstream op (torch environment).

    python torch_upstream_diff.py IN.npz OUT.npz '<case json>'

Reads the case's float32 values (bf16-exact for the bf16 cases), runs the upstream forward and
backward with the case's cotangents and writes ``o``, ``final`` and ``d<name>`` as float32 in the
layout of ``jla.ops`` (``[B, T, ...]``). Prints one JSON line with the versions.

* ``gla``: ``fla.ops.gla.chunk_gla`` (flash-linear-attention, MIT). Packed documents go in as
  ``cu_seqlens``; jla's initial state enters the row's first document (FLA's other sequences start
  at zero) and its final state is the last document's, so only that one gets a cotangent.
* ``mamba1``: ``selective_scan_cuda.fwd`` / ``.bwd`` of state-spaces/mamba (Apache-2.0), called as
  ``selective_scan_fn`` calls them (``u, delta, z [B, D, T]``, ``B, C [B, 1, N, T]``,
  ``delta_softplus=True``); the last state is ``x[:, :, -1, 1::2]`` and is not differentiated.
* ``mlstm``: NX-AI ``mlstm_kernels`` (``chunkwise--triton_xl_chunk``, chunk 128) through its public API
  only (it is not permissively licensed and is no dependency of jla): ``q, k, v [B, H, T, D]``, gate
  pre-activations ``i, f [B, H, T]``, no initial or last states.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.25)
DEV = "cuda"


def _t(data: dict, name: str, dtype, grad: bool = True) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(data[name])).to(DEV).to(dtype)
    return x.requires_grad_(grad)


def _dist_version(name: str) -> str:
    """The installed distribution's version (without importing it), or ``"?"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "?"


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def gla(data: dict, meta: dict, dtype) -> tuple[dict, dict]:
    import fla
    from fla.ops.gla import chunk_gla

    names = ["q", "k", "v", "g"] + (["h0"] if "h0" in data else [])
    x = {n: _t(data, n, dtype if n in ("q", "k", "v") else torch.float32) for n in names}
    do = torch.from_numpy(data["do"]).to(DEV)
    dht = torch.from_numpy(data["dht"]).to(DEV)
    kw = dict(q=x["q"], k=x["k"], v=x["v"], g=x["g"], scale=meta["K"] ** -0.5, output_final_state=True)
    h0 = x.get("h0")
    if "docs" in meta:
        cu = np.concatenate([[0], np.cumsum(meta["docs"])])
        kw["cu_seqlens"] = torch.tensor(cu, dtype=torch.long, device=DEV)
        if h0 is not None:
            rest = torch.zeros_like(h0).expand(len(meta["docs"]) - 1, -1, -1, -1)
            h0 = torch.cat([h0, rest])
        keep = torch.zeros(len(meta["docs"]), *dht.shape[1:], device=DEV)
        keep[-1] = dht[0]
        dht = keep
    kw["initial_state"] = h0
    o, ht = chunk_gla(**kw)
    loss = (o.float() * do).sum() + (ht.float() * dht).sum()
    grads = torch.autograd.grad(loss, [x[n] for n in names])
    out = {"o": _np(o), "final": _np(ht[-1:] if "docs" in meta else ht)}
    for n, g in zip(names, grads, strict=True):
        out["d" + n] = _np(g[:1] if n == "h0" and "docs" in meta else g)
    return out, {"fla": getattr(fla, "__version__", "?")}


def mamba1(data: dict, meta: dict, dtype) -> tuple[dict, dict]:
    import selective_scan_cuda

    def btd_to_bdt(name: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.swapaxes(data[name], 1, 2))).to(DEV).to(dtype).contiguous()

    u, delta, z = btd_to_bdt("v"), btd_to_bdt("dt"), btd_to_bdt("z")
    b = btd_to_bdt("k").unsqueeze(1).contiguous()  # [B, 1, N, T]
    c = btd_to_bdt("q").unsqueeze(1).contiguous()
    a = torch.from_numpy(data["a"]).to(DEV).contiguous()
    d = torch.from_numpy(data["d"]).to(DEV).contiguous()
    bias = torch.from_numpy(data["dt_bias"]).to(DEV).contiguous()
    out, xs, *rest = selective_scan_cuda.fwd(u, delta, a, b, c, d, z, bias, True)
    out_z = rest[0]
    last = xs[:, :, -1, 1::2]
    dout = torch.from_numpy(np.ascontiguousarray(np.swapaxes(data["do"], 1, 2))).to(DEV).to(dtype).contiguous()
    du, ddelta, da, db, dc, dd, dbias, *rest = selective_scan_cuda.bwd(
        u, delta, a, b, c, d, z, bias, dout, xs, out, None, True, False)
    dz = rest[0]

    def bdt_to_btd(x: torch.Tensor) -> np.ndarray:
        return np.swapaxes(_np(x), 1, 2)

    res = {"o": bdt_to_btd(out_z), "final": _np(last), "dq": bdt_to_btd(dc.squeeze(1)),
           "dk": bdt_to_btd(db.squeeze(1)), "dv": bdt_to_btd(du), "ddt": bdt_to_btd(ddelta), "da": _np(da),
           "dd": _np(dd), "dz": bdt_to_btd(dz), "ddt_bias": _np(dbias)}
    return res, {"mamba_ssm": _dist_version("mamba_ssm"), "kernel": "selective_scan_cuda"}


def mlstm(data: dict, meta: dict, dtype) -> tuple[dict, dict]:
    import mlstm_kernels
    from mlstm_kernels.torch import get_mlstm_kernel

    kernel = get_mlstm_kernel("chunkwise--triton_xl_chunk")

    def bhtd(name: str, grad: bool = True) -> torch.Tensor:
        x = np.ascontiguousarray(np.swapaxes(data[name], 1, 2))  # [B, T, H, ...] -> [B, H, T, ...]
        return torch.from_numpy(x).to(DEV).to(dtype).requires_grad_(grad)

    x = {n: bhtd(n) for n in ("q", "k", "v", "i", "f")}
    do = torch.from_numpy(np.ascontiguousarray(np.swapaxes(data["do"], 1, 2))).to(DEV).to(dtype)
    h = kernel(q=x["q"], k=x["k"], v=x["v"], i=x["i"], f=x["f"], return_last_states=False, chunk_size=128)
    grads = torch.autograd.grad(h, [x[n] for n in ("q", "k", "v", "i", "f")], grad_outputs=do)
    res = {"o": np.swapaxes(_np(h), 1, 2)}
    for n, g in zip(("q", "k", "v", "i", "f"), grads, strict=True):
        res["d" + n] = np.swapaxes(_np(g), 1, 2)
    return res, {"mlstm_kernels": getattr(mlstm_kernels, "__version__", "?"), "kernel": "chunkwise--triton_xl_chunk",
                 "chunk_size": 128}


def main() -> None:
    inp, outp, meta = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    data = dict(np.load(inp))
    dtype = torch.float32 if meta.get("dtype") == "float32" else torch.bfloat16
    res, info = {"gla": gla, "mamba1": mamba1, "mlstm": mlstm}[meta["op"]](data, meta, dtype)
    np.savez(outp, **res)
    print(json.dumps(dict(info, torch=torch.__version__)))


if __name__ == "__main__":
    main()
