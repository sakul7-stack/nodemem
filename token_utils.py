"""
token_utils.py -- token counting and cost utilities for the persistent
note-based memory experiments.

Design notes
------------
* Ground-truth token counts come from the Anthropic API's own tokenizer via
  ``client.messages.count_tokens`` (and from the ``usage`` field of real
  completions).  We deliberately do NOT use ``tiktoken`` as the primary
  counter: ``tiktoken`` is OpenAI's tokenizer and undercounts Claude tokens by
  ~15-20%, which would bias every metric in the paper.  A character-based
  estimator is provided only as an offline fallback and is clearly labelled as
  approximate.

* The cost model includes terms the paper's Eq. (1)-(2) omit: the small-model
  extractor cost and the cost of re-sending the (growing) notes file.  Those
  terms are what make the honest accounting different from the projection.

Prices are USD per 1,000 tokens (input, output), taken from the Anthropic
pricing table current at time of writing.  Override via ``PRICES`` if they
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


# USD per 1,000 tokens: (input, output).  Source: Anthropic pricing.
#   Haiku 4.5 : $1 / $5  per 1M  -> 0.0010 / 0.0050 per 1k
#   Sonnet 4.6: $3 / $15 per 1M  -> 0.0030 / 0.0150 per 1k
#   Opus 4.8  : $5 / $25 per 1M  -> 0.0050 / 0.0250 per 1k
PRICES = {
    "claude-haiku-4-5": (0.0010, 0.0050),
    "claude-sonnet-4-6": (0.0030, 0.0150),
    "claude-opus-4-8": (0.0050, 0.0250),
    # Groq (published per-1M rates / 1000): small extractor vs large synthesizer
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "openai/gpt-oss-120b": (0.00015, 0.00075),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.00011, 0.00034),
    # Nara Router / Mistral (USD per 1k tokens)
    "mimo-v2.5-free": (0.0, 0.0),
    "mistral-large": (0.0020, 0.0060),
    "mistral-large-latest": (0.0020, 0.0060),
    "mistral-small-latest": (0.0002, 0.0006),
    "open-mistral-nemo": (0.00015, 0.00015),
}


def price_for(model: str) -> tuple[float, float]:
    """Return (input_per_1k, output_per_1k) for a model id, with a sane default."""
    if model in PRICES:
        return PRICES[model]
    # Unknown model: assume Sonnet-tier pricing so cost is never silently zero.
    return PRICES["claude-sonnet-4-6"]


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Cost in USD for a single call given input/output token counts."""
    cin, cout = price_for(model)
    return (input_tokens / 1000.0) * cin + (output_tokens / 1000.0) * cout


def estimate_tokens_offline(text: str) -> int:
    """Approximate token count without an API call.

    Uses ~4 characters/token, the standard rough heuristic.  ONLY for offline
    plumbing tests -- never reported as a measured result.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4.0))


_TT_ENC = None


def _tiktoken_enc():
    global _TT_ENC
    if _TT_ENC is None:
        try:
            import tiktoken
            _TT_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TT_ENC = False
    return _TT_ENC


def count_tokens_local(text: str) -> int:
    """Consistent local token count via tiktoken (cl100k).

    NOTE: cl100k is OpenAI's tokenizer; absolute counts differ from Claude's by
    ~15-20%.  We use it only where the same tokenizer is applied to every
    strategy, so the *relative* token-reduction percentage is preserved (the
    ~constant factor cancels in the ratio).  Reported counts are labelled as
    tokenizer-estimated, distinct from the real Opus ``usage`` counts.
    """
    if not text:
        return 0
    enc = _tiktoken_enc()
    if enc:
        return len(enc.encode(text))
    return estimate_tokens_offline(text)


def assemble_text(system, messages: list[dict]) -> str:
    parts = []
    if system:
        parts.append(system)
    for m in messages:
        c = m["content"] if isinstance(m["content"], str) else str(m["content"])
        parts.append(f'{m["role"]}: {c}')
    return "\n".join(parts)


def count_message_tokens(
    client,
    model: str,
    system: Optional[str],
    messages: list[dict],
) -> int:
    """Ground-truth input token count for a (system, messages) prompt.

    Falls back to the offline estimator if ``client`` is None (offline mode) or
    the API call fails for any reason.
    """
    if client is None:
        joined = (system or "") + "\n".join(
            m["content"] if isinstance(m["content"], str) else str(m["content"])
            for m in messages
        )
        return estimate_tokens_offline(joined)
    try:
        kwargs = {"model": model, "messages": messages}
        if system:
            kwargs["system"] = system
        resp = client.messages.count_tokens(**kwargs)
        return int(resp.input_tokens)
    except Exception:
        joined = (system or "") + "\n".join(
            m["content"] if isinstance(m["content"], str) else str(m["content"])
            for m in messages
        )
        return estimate_tokens_offline(joined)


@dataclass
class CostLedger:
    """Accumulates token usage and cost across a run, separating the cheap
    extractor calls from the expensive synthesizer calls so the paper can report
    where the money actually goes.
    """

    synth_input_tokens: int = 0
    synth_output_tokens: int = 0
    extract_input_tokens: int = 0
    extract_output_tokens: int = 0
    synth_model: str = "claude-sonnet-4-6"
    extract_model: str = "claude-haiku-4-5"

    def add_synth(self, input_tokens: int, output_tokens: int = 0) -> None:
        self.synth_input_tokens += input_tokens
        self.synth_output_tokens += output_tokens

    def add_extract(self, input_tokens: int, output_tokens: int = 0) -> None:
        self.extract_input_tokens += input_tokens
        self.extract_output_tokens += output_tokens

    @property
    def synth_cost(self) -> float:
        return cost_usd(self.synth_model, self.synth_input_tokens, self.synth_output_tokens)

    @property
    def extract_cost(self) -> float:
        return cost_usd(self.extract_model, self.extract_input_tokens, self.extract_output_tokens)

    @property
    def total_cost(self) -> float:
        return self.synth_cost + self.extract_cost

    @property
    def total_input_tokens(self) -> int:
        return self.synth_input_tokens + self.extract_input_tokens

    def as_dict(self) -> dict:
        return {
            "synth_input_tokens": self.synth_input_tokens,
            "synth_output_tokens": self.synth_output_tokens,
            "extract_input_tokens": self.extract_input_tokens,
            "extract_output_tokens": self.extract_output_tokens,
            "synth_cost_usd": round(self.synth_cost, 6),
            "extract_cost_usd": round(self.extract_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
        }


def reduction_pct(baseline: float, value: float) -> float:
    """Percent reduction of ``value`` relative to ``baseline`` (e.g. token reduction)."""
    if baseline <= 0:
        return 0.0
    return (baseline - value) / baseline * 100.0


def fraction_of_baseline_pct(baseline: float, value: float) -> float:
    """``value`` expressed as a percentage OF ``baseline`` (the convention we use
    for the 'Cost' column so it is unambiguous)."""
    if baseline <= 0:
        return 0.0
    return value / baseline * 100.0
