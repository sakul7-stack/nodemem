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
    ``Hierarchical``, ``SlotMemory``) sharing a common ``build_context`` interface so the
  evaluator can treat them uniformly.

Every strategy answers a query against a *conversation history* (a list of
``{"role", "content"}`` messages) by producing the ``(system, messages)`` it
would actually send to the large model.  That is what makes token accounting
apples-to-apples across methods.
"""

from __future__ import annotations

import json
import math
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


class BudgetExhaustedError(RuntimeError):
    """Raised when every key is rate-limited for longer than a single call is
    willing to wait -- typically a hard daily-token (TPD) cap rather than a
    transient per-minute throttle. Signals the harness to checkpoint and stop
    cleanly so the run can be continued later with ``--resume`` once quota
    refreshes, instead of hanging forever or recording quota failures as wrong
    answers (which would corrupt retention numbers)."""


def _retry_after_s(exc, default: float) -> float:
    """Best-effort extraction of a Retry-After delay (seconds) from a 429.

    Groq returns either a ``Retry-After`` header (seconds) or, sometimes, only
    an error body like ``... try again in 8.5s``. Fall back to ``default`` when
    neither is parseable. A small margin is added so we don't wake up a hair
    before the quota window actually rolls over.
    """
    try:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) or {}
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            return float(ra) + 0.5
        msg = str(getattr(exc, "message", "") or exc)
        m = re.search(r"try again in ([\d.]+)s", msg)
        if m:
            return float(m.group(1)) + 0.5
    except (ValueError, TypeError, AttributeError):
        pass
    return default


class OpenAICompatClient:
    """OpenAI-compatible backend (e.g. Groq) exposing the same ``complete``
    interface and ``Completion`` shape as ``LLMClient``.

    Lets the *same* harness run a true cheap/expensive cascade when a single
    provider offers multiple model sizes (small extractor + large synthesizer).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 min_interval_s: float = 0.0, max_retries: int = 0,
                 api_keys: Optional[list[str]] = None, timeout_s: float = 30.0):
        from openai import OpenAI

        key_candidates: list[str] = []
        if api_keys:
            key_candidates.extend(api_keys)
        elif api_key:
            key_candidates.append(api_key)
        elif os.environ.get("OPENAI_API_KEY"):
            key_candidates.append(os.environ["OPENAI_API_KEY"])
        elif os.environ.get("GROQ_API_KEY"):
            key_candidates.append(os.environ["GROQ_API_KEY"])
        elif os.environ.get("GROQ_API_KEYS"):
            key_candidates.extend(os.environ["GROQ_API_KEYS"].split(","))

        key_candidates = [k.strip() for k in key_candidates if k and k.strip()]
        if not key_candidates:
            raise ValueError("OpenAI-compatible client requires an API key")

        # An explicit per-request timeout prevents a single hung request from
        # stalling a whole run behind the SDK's long default timeout.
        self._clients = [
            OpenAI(
                api_key=key,
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                max_retries=max_retries,
                timeout=timeout_s,
            )
            for key in key_candidates
        ]
        self._client_index = 0
        self._min_interval = min_interval_s
        self._last_call = 0.0
        # Per-key cooldown: after a key returns 429, we skip it until this wall
        # clock time. On the Groq free tier the 70B-class judge models are
        # capped near ~1 RPM per key, so hammering a just-throttled key every
        # 0.2s guarantees it stays 429'd. Parking it for a cooldown window lets
        # the key's quota actually recover, turning aggregate throughput from
        # ~0 into ~(num_keys / cooldown) RPM.
        self._cooldown_s = float(os.environ.get("GROQ_KEY_COOLDOWN_S", "12"))
        self._key_ready_at = [0.0] * len(self._clients)
        # Give up a single call after this many seconds of every key being
        # throttled -- distinguishes a transient per-minute throttle (worth
        # waiting out) from a hard daily-token exhaustion (raise and let the
        # harness checkpoint + stop for --resume). 0 disables the deadline.
        self._call_deadline_s = float(os.environ.get("GROQ_CALL_DEADLINE_S", "300"))

    @property
    def raw(self):
        return self._clients[self._client_index]

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
        max_attempts = 10000
        call_start = time.time()
        for attempt in range(max_attempts):
            nc = len(self._clients)
            # Round-robin from the current cursor, but skip keys still cooling
            # down from a recent 429. Pick the first ready key; if none are
            # ready, sleep until the soonest one is (never busy-spin).
            now = time.time()
            idx = None
            for step in range(nc):
                cand = (self._client_index + step) % nc
                if self._key_ready_at[cand] <= now:
                    idx = cand
                    break
            if idx is None:
                # Every key is throttled. If we've been stuck past the deadline,
                # this is almost certainly a hard daily cap, not a transient
                # blip: stop so the harness can checkpoint and resume later.
                if self._call_deadline_s and (now - call_start) > self._call_deadline_s:
                    raise BudgetExhaustedError(
                        f"all {nc} keys rate-limited for >{self._call_deadline_s:.0f}s "
                        f"(model={model}); likely daily-token exhaustion")
                soonest = min(self._key_ready_at)
                time.sleep(max(0.05, min(soonest - now, self._call_deadline_s or 1e9)))
                continue
            self._client_index = (idx + 1) % nc
            client = self._clients[idx]
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, messages=oai_msgs)
                break
            except (openai.AuthenticationError, openai.PermissionDeniedError):
                if len(self._clients) <= 1:
                    raise
                self._clients.pop(idx)
                self._key_ready_at.pop(idx)
                self._client_index = self._client_index % len(self._clients)
                time.sleep(0.5)
            except openai.RateLimitError as e:
                # Park this key so its quota recovers instead of re-hitting it
                # on the next cycle. Prefer the server's Retry-After hint when
                # present; otherwise fall back to the fixed cooldown window.
                self._key_ready_at[idx] = time.time() + _retry_after_s(e, self._cooldown_s)
            except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError):
                time.sleep(1)
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
        if system and "fact slots" in (system or "").lower():
            text = json.dumps({"facts": [asdict(f) for f in parse_slot_facts(user_text)]})
        elif system and "extract" in (system or "").lower():
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

    def __post_init__(self) -> None:
        self.entities = _as_text_list(self.entities)
        self.decisions = _as_text_list(self.decisions)
        self.facts = _as_text_list(self.facts)
        self.goals = _as_text_list(self.goals)
        self.action_items = _as_text_list(self.action_items)


def _as_text_list(value) -> list[str]:
    """Convert extractor output into a robust list of short strings.

    Real models occasionally return objects inside arrays despite the prompt.
    Rendering must stay total so one malformed note cannot drop an entire case.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = "; ".join(f"{k}: {v}" for k, v in item.items())
        else:
            text = str(item)
        if text:
            out.append(text)
    return out


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


# --------------------------------------------------------------------------- #
# Fact-slot memory (query-addressable, latest-value table)
# --------------------------------------------------------------------------- #


@dataclass
class SlotFact:
    timestamp: int
    segment_id: int
    subject: str
    predicate: str
    value: str
    evidence: str = ""

    def __post_init__(self) -> None:
        self.subject = str(self.subject).strip()
        self.predicate = str(self.predicate).strip()
        self.value = str(self.value).strip()
        self.evidence = str(self.evidence).strip()

    @property
    def key(self) -> str:
        return _slot_key(self.subject, self.predicate)


def _slot_key(subject: str, predicate: str) -> str:
    text = f"{subject}::{predicate}".lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _slot_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def parse_slot_facts(text: str, timestamp: int = 0, segment_id: int = 0) -> list[SlotFact]:
    """Parse the synthetic fact grammar into latest-value slots.

    This is intentionally narrow: it is a deterministic baseline for the
    released synthetic suite, not a general information extractor. Live runs can
    populate the same slot schema with an LLM extractor.
    """
    patterns = [
        ("budget", r"approved budget for (?P<subject>.+?) (?:is|has been updated to) \$?(?P<value>[0-9,]+)"),
        ("deadline", r"hard deadline for (?P<subject>.+?) (?:is|has been updated to) (?P<value>[A-Za-z]+\s+\d+)"),
        ("owner", r"owner of (?P<subject>.+?) (?:is|has been updated to) (?P<value>[A-Za-z]+)"),
        ("target metric", r"target metric for (?P<subject>.+?) (?:is|has been updated to) (?P<value>\d+) percent"),
        ("hard constraint", r"hard constraint on (?P<subject>.+?) (?:is that it must|has been updated to that it must) (?P<value>[^.\n]+)"),
    ]
    facts: list[SlotFact] = []
    for predicate, pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            subject = match.group("subject").strip(" .:\n")
            # Strip role/preamble fragments that can precede the fact sentence.
            subject = re.sub(r"^.*?(?:for|of|on)\s+", "", subject, flags=re.I)
            value = match.group("value").strip(" .")
            evidence = match.group(0).strip()
            facts.append(SlotFact(timestamp, segment_id, subject, predicate, value, evidence))
    return facts


class SlotMemoryStore:
    """Latest-value table for durable facts.

    Unlike append-only notes, this store updates one slot per subject/predicate
    and renders only compact current values. It is therefore designed to test
    whether the memory representation, not just the extractor model, is the
    bottleneck in the pilot results.
    """

    def __init__(self):
        self.slots: dict[str, SlotFact] = {}
        self.events: list[SlotFact] = []

    def upsert(self, fact: SlotFact) -> None:
        if not fact.subject or not fact.predicate or not fact.value:
            return
        self.events.append(fact)
        current = self.slots.get(fact.key)
        if current is None or fact.timestamp >= current.timestamp:
            self.slots[fact.key] = fact

    def ingest_text(self, text: str, timestamp: int, segment_id: int) -> None:
        for fact in parse_slot_facts(text, timestamp, segment_id):
            self.upsert(fact)

    def ingest_data(self, data: dict, timestamp: int, segment_id: int) -> None:
        if isinstance(data, list):
            data = {"facts": data}
        if not isinstance(data, dict):
            return
        for item in data.get("facts", []) or []:
            if isinstance(item, dict):
                subject = item.get("subject") or item.get("entity") or item.get("name") or ""
                predicate = item.get("predicate") or item.get("attribute") or item.get("key") or ""
                value = item.get("value") or item.get("answer") or item.get("current_value") or ""
                evidence = item.get("evidence") or item.get("text") or item.get("quote") or ""
                self.upsert(SlotFact(timestamp, segment_id, subject, predicate, value, evidence))
            elif isinstance(item, str):
                for fact in parse_slot_facts(item, timestamp, segment_id):
                    self.upsert(fact)

    def facts_for_segment(self, segment_id: int) -> list[SlotFact]:
        return [fact for fact in self.events if fact.segment_id == segment_id]

    def render(self, query: str = "", max_slots: int = 24) -> str:
        facts = list(self.slots.values())
        if query:
            q = _slot_tokens(query)
            facts.sort(
                key=lambda f: (
                    len(q & _slot_tokens(f"{f.subject} {f.predicate} {f.value} {f.evidence}")),
                    f.timestamp,
                ),
                reverse=True,
            )
        else:
            facts.sort(key=lambda f: (f.timestamp, f.subject, f.predicate), reverse=True)
        if max_slots:
            facts = facts[:max_slots]
        lines = [
            f"- {fact.subject} | {fact.predicate} | {fact.value} (t={fact.timestamp})"
            for fact in facts
        ]
        return "\n".join(lines)


EXTRACT_SYSTEM = (
    "You extract structured memory from a conversation segment. "
    "Return ONLY a JSON object with keys: entities (people/orgs/products), "
    "decisions, facts (include any numbers, dates, IDs verbatim), goals, "
    "action_items. Each value is a list of short strings. Be faithful and "
    "concise; do not invent information."
)

SLOT_EXTRACT_SYSTEM = (
    "You extract durable fact slots from a conversation segment. Return ONLY "
    "JSON with key facts, where facts is a list of objects: subject, predicate, "
    "value, evidence. Use one slot for each stable fact such as budget, owner, "
    "deadline, metric, constraint, preference, or decision. If a segment corrects "
    "a fact, emit the corrected current value. Be literal and concise."
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


class SlotMemory(Strategy):
    """Structured latest-value memory: fact slots + recent window."""

    name = "slot_memory"

    def __init__(self, recent_k: int = 6, store: Optional[SlotMemoryStore] = None,
                 max_slots: int = 24):
        self.recent_k = recent_k
        self.store = store if store is not None else SlotMemoryStore()
        self.max_slots = max_slots

    def build_context(self, history, query):
        rendered = self.store.render(query=query, max_slots=self.max_slots)
        system = (
            SYNTH_SYSTEM_BASE
            + "\n\nCurrent structured fact slots (latest values only):\n"
            + (rendered or "(none)")
        )
        msgs = history[-self.recent_k:] + [{"role": "user", "content": query}]
        return system, msgs


# --------------------------------------------------------------------------- #
# Retrieval-augmented memory (sparse BM25 by default, optional dense embeddings)
# --------------------------------------------------------------------------- #


def _retr_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Minimal in-memory BM25 (Okapi) over conversation chunks.

    Pure Python, no dependencies. BM25 is the sparse-retrieval baseline used by
    LongMemEval, so this is a faithful, reproducible retrieval comparator.
    """

    backend = "bm25"

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tokenized = [_retr_tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.tokenized) / self.N if self.N else 0.0
        df: dict[str, int] = {}
        for toks in self.tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        q = _retr_tokens(query)
        scored: list[tuple[float, int]] = []
        for i, toks in enumerate(self.tokenized):
            if not toks:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            score = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                score += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.docs[i] for _, i in scored[:top_k]]


class DenseRetriever:
    """Optional dense (embedding) retriever via sentence-transformers.

    Used only when ``backend='dense'`` and the dependency is installed; otherwise
    the harness falls back to BM25 so the default install stays light.
    """

    backend = "dense"

    def __init__(self, docs: list[str], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy
        import numpy as np

        self._np = np
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(
            docs or [""], normalize_embeddings=True, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self.docs:
            return []
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.emb @ q
        idx = self._np.argsort(-sims)[:top_k]
        return [self.docs[i] for i in idx]


def _chunk_case(case, granularity: str = "turn") -> list[str]:
    """Turn the conversation into retrievable text chunks."""
    msgs = case.messages
    if granularity == "turn":
        return [m["content"] for m in msgs if m.get("content")]
    # session: group consecutive turns until a new "[Session" header appears
    chunks, cur = [], []
    for m in msgs:
        c = m.get("content", "")
        if c.startswith("[Session") and cur:
            chunks.append("\n".join(cur))
            cur = []
        cur.append(c)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def build_retriever(case, backend: str = "bm25", granularity: str = "turn"):
    """Build a retriever over a conversation. Falls back to BM25 if the dense
    backend is requested but unavailable."""
    docs = _chunk_case(case, granularity)
    if backend == "dense":
        try:
            return DenseRetriever(docs)
        except Exception:
            return BM25Retriever(docs)
    return BM25Retriever(docs)


class RetrievalRAG(Strategy):
    """Retrieval-augmented baseline: fetch the top-k most relevant past turns for
    the query and inject them alongside the recent window."""

    name = "rag"

    def __init__(self, recent_k: int = 6, retriever=None, top_k: int = 5):
        self.recent_k = recent_k
        self.retriever = retriever
        self.top_k = top_k

    def build_context(self, history, query):
        chunks = self.retriever.retrieve(query, self.top_k) if self.retriever else []
        mem = "\n".join(f"- {c}" for c in chunks) or "(none)"
        system = (SYNTH_SYSTEM_BASE
                  + "\n\nRetrieved relevant memory (top matches for this query):\n" + mem)
        msgs = history[-self.recent_k:] + [{"role": "user", "content": query}]
        return system, msgs


class SlotRAGHybrid(Strategy):
    """Hybrid memory: structured latest-value slots for durable facts PLUS
    retrieved narrative turns for everything the slots do not capture."""

    name = "slot_rag_hybrid"

    def __init__(self, recent_k: int = 6, store: Optional[SlotMemoryStore] = None,
                 retriever=None, top_k: int = 5, max_slots: int = 24):
        self.recent_k = recent_k
        self.store = store if store is not None else SlotMemoryStore()
        self.retriever = retriever
        self.top_k = top_k
        self.max_slots = max_slots

    def build_context(self, history, query):
        slots = self.store.render(query=query, max_slots=self.max_slots) or "(none)"
        chunks = self.retriever.retrieve(query, self.top_k) if self.retriever else []
        mem = "\n".join(f"- {c}" for c in chunks) or "(none)"
        system = (SYNTH_SYSTEM_BASE
                  + "\n\nCurrent structured fact slots (latest values only):\n" + slots
                  + "\n\nRetrieved relevant turns:\n" + mem)
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
