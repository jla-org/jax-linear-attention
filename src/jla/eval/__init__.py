# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation: per-token scoring, perplexity (``jla eval ppl``) and the lm-eval adapter (``jla eval lm-eval``).

``jla.eval.lm_eval_adapter`` imports the optional ``lm-eval`` package, so it is not imported here.
"""

from jla.eval.perplexity import PerplexityResult, perplexity, stream_windows
from jla.eval.scoring import score_sequences, token_logprobs

__all__ = ["PerplexityResult", "perplexity", "score_sequences", "stream_windows", "token_logprobs"]
