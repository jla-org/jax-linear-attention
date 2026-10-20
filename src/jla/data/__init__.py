# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token data: CIDX memory-mapped corpora, datasets, a forked-worker loader and tokenization.

* :mod:`jla.data.cidx`: the CIDX format (``.idx`` index + ``.bin`` tokens);
* :mod:`jla.data.datasets`: fixed-length windows, weighted blends, synthetic data;
* :mod:`jla.data.loader`: deterministic, resumable batch loader;
* :mod:`jla.data.packing`: ``segment_ids``/``positions`` from EOS boundaries;
* :mod:`jla.data.pipeline`: config-driven sources, data state, device prefetch;
* :mod:`jla.data.make_data`: JSONL to CIDX tokenizer CLI;
* :mod:`jla.data.streaming`: raw-text streams tokenized, packed, sharded and resumed on the fly;
* :mod:`jla.data.sft`: SFT corpora (CIDX + loss-mask sidecar) packed into document-masked rows;
* :mod:`jla.data.sft_tokenize`: chat-template / instruction JSONL to SFT corpus CLI.
"""
