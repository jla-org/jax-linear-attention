# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FLA's ``chunk_kda`` on the inputs of one ``diff_vector_gate.py`` case (run in a torch environment).

    python torch_kda_diff.py IN.npz OUT.npz '<case json>'

Reads the float32 values of the case (bf16-exact where the case is bf16), runs ``fla.ops.kda.chunk_kda``
forward and backward with the cotangents of the case and writes ``o``, ``final`` and ``d<name>`` as float32.
Packed documents go in as ``cu_seqlens``; ``final`` is then the last sequence's state (jla's final state of
the row) and only its cotangent is non-zero. The raw-gate cases use FLA's in-kernel gate, beta sigmoid and
L2 norm. Prints one JSON line with the FLA version.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.25)


def main() -> None:
    inp, outp, meta = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    import fla
    from fla.ops.kda import chunk_kda

    dev = "cuda"
    data = dict(np.load(inp))
    dtype = torch.float32 if meta.get("dtype") == "float32" else torch.bfloat16

    def tensor(name, dt=torch.float32):
        return torch.from_numpy(data[name]).to(dev).to(dt).requires_grad_(True)

    names = meta["names"]
    x = {n: tensor(n, dtype if n in ("q", "k", "v") else torch.float32) for n in names}
    kw = dict(q=x["q"], k=x["k"], v=x["v"], g=x["g"], beta=x["beta"], scale=meta["K"] ** -0.5,
              initial_state=x.get("h0"), output_final_state=True,
              use_qk_l2norm_in_kernel=bool(meta.get("l2norm")))
    if meta["gate"] == "lower_bound":
        kw.update(use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, A_log=x["A_log"],
                  dt_bias=x["dt_bias"], use_beta_sigmoid_in_kernel=True)
    elif meta["gate"] == "softplus":
        kw.update(use_gate_in_kernel=True, A_log=x["A_log"], dt_bias=x["dt_bias"], use_beta_sigmoid_in_kernel=True)
    do = torch.from_numpy(data["do"]).to(dev)
    dht = torch.from_numpy(data["dht"]).to(dev)
    if "cu" in meta:
        kw["cu_seqlens"] = torch.tensor(meta["cu"], dtype=torch.long, device=dev)
        keep = torch.zeros_like(dht)
        keep[-1] = dht[-1]
        dht = keep
        if "h0" in x:  # jla's initial state enters the row's first document; FLA's other sequences start at zero
            rest = torch.zeros_like(x["h0"]).expand(len(meta["cu"]) - 2, -1, -1, -1)
            kw["initial_state"] = torch.cat([x["h0"], rest])
    o, ht = chunk_kda(**kw)
    loss = (o.float() * do).sum() + (ht.float() * dht).sum()
    grads = torch.autograd.grad(loss, [x[n] for n in names])
    final = ht[-1:] if "cu" in meta else ht
    out = dict(o=o.detach().float().cpu().numpy(), final=final.detach().float().cpu().numpy())
    out.update({f"d{n}": gr.detach().float().cpu().numpy() for n, gr in zip(names, grads)})
    if meta.get("flashkda"):  # FlashKDA's fused forward on its own contract (bf16 raw gate and beta logits)
        import flash_kda

        H, K = x["q"].shape[-2:]
        o_fk = torch.empty_like(x["v"])
        with torch.inference_mode():
            flash_kda.fwd(x["q"].detach(), x["k"].detach(), x["v"].detach(), x["g"].detach().bfloat16(),
                          x["beta"].detach().bfloat16(), meta["K"] ** -0.5, o_fk, x["A_log"].detach(),
                          x["dt_bias"].detach().view(H, K), -5.0, cu_seqlens=kw.get("cu_seqlens"))
        out["o_flashkda"] = o_fk.float().cpu().numpy()
    np.savez(outp, **out)
    print(json.dumps(dict(fla=fla.__version__, torch=torch.__version__, case=meta.get("case"))))


if __name__ == "__main__":
    main()
