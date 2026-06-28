"""
hierarchical_memory.py -- reference implementation of persistent note-based
memory and the comparison baselines.

Public surface
--------------
* ``LLMClient``      : thin wrapper over the Anthropic SDK (small + large model).
* ``MockLLMClient``  : deterministic, offline stand-in for plumbing/CI.
* ``Note`` / ``NotesFile`` : the append-only, conflict-aware notes store.
* ``HierarchicalMemory`` : the proposed method (cheap extractor + notes + recent).
* Strategy classes (``FullContext``, ``Truncation``, ``RunningSummary``,
  ``Hierarchical``) sharing a common ``build_context`` interface so the
  evaluator can treat them uniformly.

Every strategy answers a query against a *conversation history* (a list of
``{"role", "content"}`` messages) by producing the ``(system, messages)`` it
would actually send to the large model.  That is what makes token accounting
apples-to-apples across methods.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- #
# Provider abstraction
# --------------------------------------------------------------------------- #


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient:
    """Wraps the Anthropic SDK. Exposes one ``complete`` call plus token counting.

    The same client object backs both the small extractor and the large
    synthesizer; the model id is passed per call.
    """

    def __init__(self, api_key: Optional[str] = None, min_interval_s: float = 0.0,
                 max_retries: int = 1):
        import anthropic  # imported lazily so offline mode needs no SDK

        # We handle 429/503 backoff ourselves (see complete()); keep the SDK's
        # internal retries low so they don't burst past a per-minute cap.
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
        )
        self._min_interval = min_interval_s   # throttle for rate-limited proxies
        self._last_call = 0.0

    @property
    def raw(self):
        return self._client

    def _throttle(self):
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def complete(
        self,
        model: str,
        system: Optional[str],
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> Completion:
        import anthropic

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        # Hard per-minute caps on shared proxies surface as 429s that outlast the
        # SDK's exponential backoff; sleep out the window and retry.
        for attempt in range(8):
            self._throttle()
            t0 = time.time()
            try:
                resp = self._client.messages.create(**kwargs)
                break
            except anthropic.RateLimitError:
                time.sleep(40)
            except anthropic.InternalServerError:
                time.sleep(5)
        else:
            raise RuntimeError("exhausted retries against rate-limited endpoint")
        latency = time.time() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if resp.usage is None:
            # Some proxies omit usage; fall back to a local estimate so the run
            # never silently records zero-cost calls.
            from token_utils import count_tokens_local, assemble_text
            in_tok = count_tokens_local(assemble_text(system, messages))
            out_tok = count_tokens_local(text)
        else:
            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
        return Completion(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
        )


class OpenAICompatClient:
    """OpenAI-compatible backend (e.g. Groq) exposing the same ``complete``
    interface and ``Completion`` shape as ``LLMClient``.

    Lets the *same* harness run a true cheap/expensive cascade when a single
    provider offers multiple model sizes (small extractor + large synthesizer).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 min_interval_s: float = 0.0, max_retries: int = 1):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            max_retries=max_retries,
        )
        self._min_interval = min_interval_s
        self._last_call = 0.0

    @property
    def raw(self):
        return self._client

    def _throttle(self):
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def complete(self, model, system, messages, max_tokens: int = 1024) -> Completion:
        import openai

        oai_msgs = ([{"role": "system", "content": system}] if system else []) + messages
        for attempt in range(15):
            self._throttle()
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(
                    model=model, max_tokens=max_tokens, messages=oai_msgs)
                break
            except openai.RateLimitError as e:
                # Honor Retry-After when present (Groq returns it for TPM limits).
                ra = None
                try:
                    ra = float(e.response.headers.get("retry-after", "")) if e.response else None
                except Exception:
                    ra = None
                time.sleep(min(max(ra or 20.0, 8.0), 65.0))
            except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError):
                time.sleep(6)
        else:
            raise RuntimeError("exhausted retries against rate-limited endpoint")
        latency = time.time() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        if u is None:
            from token_utils import count_tokens_local, assemble_text
            in_tok = count_tokens_local(assemble_text(system, messages))
            out_tok = count_tokens_local(text)
        else:
            in_tok, out_tok = u.prompt_tokens, u.completion_tokens
        return Completion(text=text, input_tokens=in_tok, output_tokens=out_tok,
                          latency_s=latency)


class MockLLMClient:
    """Deterministic offline client.

    * ``complete`` for extraction returns a JSON note built by a heuristic
      (entities/decisions/facts pulled from the text).
    * ``complete`` for synthesis returns a keyword-overlap "answer" so the
      end-to-end harness runs without an API key.  Mock answers are NOT
      scientific results -- they exist to test plumbing and to compute the
      deterministic token-accounting metrics.
    """

    def __init__(self):
        self._client = None

    @property
    def raw(self):
        return None

    def complete(self, model, system, messages, max_tokens: int = 1024) -> Completion:
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        if system and "extract" in (system or "").lower():
            note = _heuristic_extract(user_text)
            text = json.dumps(note)
        else:
            # crude extractive "answer": echo the most fact-like sentence
            sents = re.split(r"(?<=[.!?])\s+", user_text)
            text = max(sents, key=len) if sents else ""
        from token_utils import estimate_tokens_offline

        return Completion(
            text=text,
            input_tokens=estimate_tokens_offline((system or "") + user_text),
            output_tokens=estimate_tokens_offline(text),
            latency_s=0.0,
        )


def _heuristic_extract(text: str) -> dict:
    """Offline note extraction: regex for entities/numbers/decisions."""
    entities = sorted(set(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)))[:10]
    facts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if re.search(r"\d", s)][:8]
    decisions = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if re.search(r"\b(decide|will|chose|agreed|plan to)\b", s, re.I)][:8]
    return {
        "entities": entities,
        "decisions": decisions,
        "facts": facts,
        "goals": [],
        "action_items": [],
    }


# --------------------------------------------------------------------------- #
# Notes store (conflict-aware, append-only with timestamp precedence)
# --------------------------------------------------------------------------- #


@dataclass
class Note:
    timestamp: int
    segment_id: int
    entities: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


class NotesFile:
    """Append-only notes with a render that respects recency.

    Newer notes are rendered last and prefixed so the synthesizer treats later
    facts as overriding earlier ones -- the paper's 'timestamp-based precedence'
    mitigation, made concrete.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.notes: list[Note] = []
        if path and os.path.exists(path):
            self.load()

    def append(self, note: Note) -> None:
        self.notes.append(note)
        if self.path:
            self.save()

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(n) for n in self.notes], f, indent=2)

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            self.notes = [Note(**d) for d in json.load(f)]

    def compact(self, max_notes: int = 8) -> bool:
        """Bound notes growth: once there are more than ``max_notes`` segments,
        fold the oldest half into a single consolidated note.

        This is the concrete fix for the paper's 'file grows indefinitely'
        limitation -- without it, re-sending the notes file makes the
        hierarchical method *also* O(k^2) over a long conversation.  Entities
        are de-duplicated and the most recent facts/decisions are kept, so later
        (corrective) facts survive consolidation.
        """
        if len(self.notes) <= max_notes:
            return False
        fold = self.notes[: len(self.notes) // 2]
        keep = self.notes[len(self.notes) // 2:]

        def _dedup(seq):
            seen, out = set(), []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        merged = Note(
            timestamp=fold[0].timestamp,
            segment_id=fold[0].segment_id,
            entities=_dedup(e for n in fold for e in n.entities)[:20],
            decisions=_dedup(d for n in fold for d in n.decisions)[-12:],
            facts=_dedup(f for n in fold for f in n.facts)[-16:],
            goals=_dedup(g for n in fold for g in n.goals)[:12],
            action_items=_dedup(a for n in fold for a in n.action_items)[-12:],
        )
        self.notes = [merged] + keep
        if self.path:
            self.save()
        return True

    def render(self) -> str:
        """Render notes oldest->newest as compact text for the synth context."""
        lines = []
        for n in self.notes:
            seg = [f"[segment {n.segment_id} @t={n.timestamp}]"]
            if n.entities:
                seg.append("entities: " + "; ".join(n.entities))
            if n.facts:
                seg.append("facts: " + " | ".join(n.facts))
            if n.decisions:
                seg.append("decisions: " + " | ".join(n.decisions))
            if n.goals:
                seg.append("goals: " + " | ".join(n.goals))
            if n.action_items:
                seg.append("action_items: " + " | ".join(n.action_items))
            lines.append("\n".join(seg))
        return "\n\n".join(lines)


EXTRACT_SYSTEM = (
    "You extract structured memory from a conversation segment. "
    "Return ONLY a JSON object with keys: entities (people/orgs/products), "
    "decisions, facts (include any numbers, dates, IDs verbatim), goals, "
    "action_items. Each value is a list of short strings. Be faithful and "
    "concise; do not invent information."
)

SYNTH_SYSTEM_BASE = (
    "You are an assistant answering a question using the provided memory and "
    "recent messages. If facts conflict, prefer the most recent. Answer "
    "concisely and only from the provided context."
)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


class Strategy:
    name = "base"

    def build_context(self, history: list[dict], query: str) -> tuple[Optional[str], list[dict]]:
        """Return (system, messages) the large model would receive to answer query."""
        raise NotImplementedError


class FullContext(Strategy):
    name = "full_context"

    def build_context(self, history, query):
        msgs = list(history) + [{"role": "user", "content": query}]
        return SYNTH_SYSTEM_BASE, msgs


class Truncation(Strategy):
    name = "truncation"

    def __init__(self, k: int = 10):
        self.k = k

    def build_context(self, history, query):
        msgs = history[-self.k:] + [{"role": "user", "content": query}]
        return SYNTH_SYSTEM_BASE, msgs


class RunningSummary(Strategy):
    """Single-model summarization baseline: the large model maintains a running
    summary of everything older than the recent window."""

    name = "running_summary"

    def __init__(self, recent_k: int = 6):
        self.recent_k = recent_k
        self.summary = ""

    def update_summary(self, client, model, older_messages: list[dict], ledger=None):
        if not older_messages:
            return 0.0
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in older_messages)
        prompt = (
            "Update the running summary so it preserves every entity, decision, "
            "fact (with numbers/dates), goal, and action item.\n\n"
            f"Current summary:\n{self.summary or '(none)'}\n\nNew messages:\n{convo}\n\n"
            "Return only the updated summary."
        )
        c = client.complete(model, "You maintain a faithful running summary.",
                            [{"role": "user", "content": prompt}], max_tokens=600)
        self.summary = c.text
        if ledger is not None:
            ledger.add_synth(c.input_tokens, c.output_tokens)  # same expensive model
        return c.latency_s

    def build_context(self, history, query):
        system = SYNTH_SYSTEM_BASE + "\n\nConversation summary:\n" + (self.summary or "(none)")
        msgs = history[-self.recent_k:] + [{"role": "user", "content": query}]
        return system, msgs


class Hierarchical(Strategy):
    """The proposed method: cheap extractor writes notes; synth sees notes +
    recent window."""

    name = "hierarchical"

    def __init__(self, recent_k: int = 6, notes: Optional[NotesFile] = None):
        self.recent_k = recent_k
        self.notes = notes if notes is not None else NotesFile()

    def extract_segment(self, client, extract_model, segment: list[dict], segment_id: int,
                        ledger=None) -> float:
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in segment)
        c = client.complete(extract_model, EXTRACT_SYSTEM,
                           [{"role": "user", "content": convo}], max_tokens=512)
        data = _safe_json(c.text)
        note = Note(
            timestamp=segment_id,
            segment_id=segment_id,
            entities=data.get("entities", []) or [],
            decisions=data.get("decisions", []) or [],
            facts=data.get("facts", []) or [],
            goals=data.get("goals", []) or [],
            action_items=data.get("action_items", []) or [],
        )
        self.notes.append(note)
        if ledger is not None:
            ledger.add_extract(c.input_tokens, c.output_tokens)  # CHEAP model
        return c.latency_s

    def build_context(self, history, query):
        system = (
            SYNTH_SYSTEM_BASE
            + "\n\nPersistent notes (oldest first; prefer most recent on conflict):\n"
            + (self.notes.render() or "(none)")
        )
        msgs = history[-self.recent_k:] + [{"role": "user", "content": query}]
        return system, msgs


def _safe_json(text: str) -> dict:
    """Parse model JSON, tolerating code fences / surrounding prose."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}
