"""
real_datasets.py -- adapters that load real long-conversation memory benchmarks
into the same ``Case``/``Fact`` shape used by the synthetic suite, so every
strategy in ``hierarchical_memory.py`` and the harness in ``evaluate.py`` can
run on them unchanged.

Supported benchmarks
--------------------
* **LoCoMo** (Maharana et al., ACL 2024): 10 very long multi-session
  human-like conversations (~300-690 turns each) with ~2k human-annotated QA
  pairs across five categories (single-hop, multi-hop, temporal, open-domain,
  adversarial/abstention).  Download ``data/locomo10.json`` from
  https://github.com/snap-research/locomo .

* **LongMemEval** (Wu et al., ICLR 2025): 500 questions over ~40-session,
  ~115k-token timestamped chat histories spanning information-extraction,
  multi-session reasoning, knowledge-update, temporal-reasoning, and abstention.
  Download the ``longmemeval_*`` json files from
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned .

Both datasets are openly released for research. We download them at evaluation
time and do not redistribute them in this repository.

Category semantics
------------------
The harness reports retention per ``Fact.category``. ``is_abstention`` marks
questions whose gold behavior is to decline (the information is absent); these
are scored by whether the model correctly abstains rather than by answer match.
``updated=True`` is kept for backward compatibility with the synthetic
"fact-update" probe and is set for LongMemEval ``knowledge-update`` questions.
"""

from __future__ import annotations

import json
import os
import random
import re

from datasets import Case, Fact


# --------------------------------------------------------------------------- #
# LoCoMo
# --------------------------------------------------------------------------- #

# Official LoCoMo category ids -> readable names (see snap-research/locomo).
LOCOMO_CATEGORIES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def _locomo_turn_text(turn: dict) -> str:
    """Render a LoCoMo turn as text, folding in the BLIP image caption when the
    turn shared a photo (LoCoMo provides captions in lieu of the images)."""
    text = (turn.get("text") or "").strip()
    caption = turn.get("blip_caption")
    if caption:
        text = (text + f" [shares a photo: {caption.strip()}]").strip()
    return text


def _subsample_balanced(facts: list[Fact], max_per_conv: int, seed: int) -> list[Fact]:
    """Round-robin across categories so a per-conversation cap stays balanced."""
    if not max_per_conv or len(facts) <= max_per_conv:
        return facts
    rng = random.Random(seed)
    by_cat: dict[str, list[Fact]] = {}
    for fact in facts:
        by_cat.setdefault(fact.category, []).append(fact)
    for bucket in by_cat.values():
        rng.shuffle(bucket)
    cats = sorted(by_cat)
    idx = {c: 0 for c in cats}
    out: list[Fact] = []
    while len(out) < max_per_conv and any(idx[c] < len(by_cat[c]) for c in cats):
        for c in cats:
            if idx[c] < len(by_cat[c]):
                out.append(by_cat[c][idx[c]])
                idx[c] += 1
                if len(out) >= max_per_conv:
                    break
    return out


def load_locomo(path: str = "data/locomo10.json", max_qa_per_conv: int = 0,
                seed: int = 7, categories: list[str] | None = None) -> list[Case]:
    """Load LoCoMo into ``Case`` objects (one per conversation).

    Each conversation's sessions are flattened chronologically with a session
    date header on the first turn of each session (needed for temporal
    questions). Speaker A maps to ``user`` and speaker B to ``assistant`` so the
    transcript alternates; speaker names are also kept in the message text.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LoCoMo file not found at {path}. Download it with:\n"
            "  mkdir -p data && curl -sL -o data/locomo10.json "
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases: list[Case] = []
    for sample in data:
        conv = sample["conversation"]
        speaker_a = conv.get("speaker_a")
        session_nums = sorted(
            int(m.group(1)) for k in conv if (m := re.match(r"session_(\d+)$", k))
        )
        messages: list[dict] = []
        for sid in session_nums:
            date = conv.get(f"session_{sid}_date_time", "")
            turns = conv.get(f"session_{sid}", []) or []
            first_emitted = True
            for turn in turns:
                body = _locomo_turn_text(turn)
                if not body:
                    continue
                speaker = turn.get("speaker", "")
                prefix = f"[Session {sid}, {date}] " if first_emitted else ""
                first_emitted = False
                role = "user" if speaker == speaker_a else "assistant"
                messages.append({"role": role, "content": f"{prefix}{speaker}: {body}"})

        facts: list[Fact] = []
        for qi, q in enumerate(sample.get("qa", [])):
            cat = LOCOMO_CATEGORIES.get(q.get("category"), "other")
            if categories and cat not in categories:
                continue
            key = f"{sample['sample_id']}-q{qi}"
            evidence = q.get("evidence", []) or []
            if q.get("category") == 5:
                # Adversarial: gold behavior is abstention. adversarial_answer is
                # the plausible-but-unsupported trap, stored for reference only.
                facts.append(Fact(
                    key=key, question=q["question"],
                    answer=str(q.get("adversarial_answer") or "(not answerable)"),
                    category=cat, is_abstention=True, evidence=evidence))
            else:
                if q.get("answer") is None:
                    continue
                facts.append(Fact(
                    key=key, question=q["question"], answer=str(q.get("answer", "")),
                    category=cat, evidence=evidence))

        facts = _subsample_balanced(facts, max_qa_per_conv, seed)
        cases.append(Case(id=sample["sample_id"], domain="locomo",
                          length_class=f"{len(messages)}msg",
                          messages=messages, facts=facts))
    return cases


# --------------------------------------------------------------------------- #
# LongMemEval
# --------------------------------------------------------------------------- #

# question_type -> readable category. Abstention is detected from a _abs suffix
# on the question id.
LME_CATEGORIES = {
    "single-session-user": "info_extraction",
    "single-session-assistant": "info_extraction",
    "single-session-preference": "preference",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
    "multi-session": "multi_session",
}


def load_longmemeval(path: str = "data/longmemeval_s.json", max_instances: int = 0,
                     seed: int = 7, categories: list[str] | None = None) -> list[Case]:
    """Load LongMemEval into ``Case`` objects (one per question/history pair).

    Each instance pairs a unique multi-session history with a single question.
    Sessions are flattened with a date header on the first turn of each session.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LongMemEval file not found at {path}. Download it with:\n"
            "  mkdir -p data && curl -sL -o data/longmemeval_s.json "
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_s_cleaned.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _cat_of(inst):
        qid = str(inst.get("question_id", ""))
        if qid.endswith("_abs"):
            return "abstention"
        return LME_CATEGORIES.get(inst.get("question_type", ""),
                                  inst.get("question_type", "") or "other")

    # Filter by category BEFORE sampling so a capped run hits the target category.
    if categories:
        data = [inst for inst in data if _cat_of(inst) in categories]

    rng = random.Random(seed)
    if max_instances and len(data) > max_instances:
        data = rng.sample(data, max_instances)

    cases: list[Case] = []
    for inst in data:
        qid = str(inst.get("question_id", ""))
        is_abs = qid.endswith("_abs")
        qtype = inst.get("question_type", "")
        cat = "abstention" if is_abs else LME_CATEGORIES.get(qtype, qtype or "other")
        if categories and cat not in categories:
            continue

        messages: list[dict] = []
        sessions = inst.get("haystack_sessions", []) or []
        dates = inst.get("haystack_dates", []) or []
        for si, sess in enumerate(sessions):
            date = dates[si] if si < len(dates) else ""
            for ti, turn in enumerate(sess or []):
                role = turn.get("role", "user")
                if role not in ("user", "assistant"):
                    role = "user"
                content = turn.get("content", "") or ""
                prefix = f"[Session {si + 1}, {date}] " if ti == 0 else ""
                messages.append({"role": role, "content": f"{prefix}{content}"})

        fact = Fact(
            key=qid, question=inst.get("question", ""),
            answer=str(inst.get("answer", "")),
            category=cat, is_abstention=is_abs,
            updated=(cat == "knowledge_update"))
        cases.append(Case(id=qid, domain="longmemeval",
                          length_class=f"{len(messages)}msg",
                          messages=messages, facts=[fact]))
    return cases


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #

def load_real_suite(dataset: str, **kwargs) -> list[Case]:
    if dataset == "locomo":
        return load_locomo(**kwargs)
    if dataset == "longmemeval":
        return load_longmemeval(**kwargs)
    raise ValueError(f"unknown real dataset: {dataset!r}")


if __name__ == "__main__":
    import sys
    from collections import Counter

    name = sys.argv[1] if len(sys.argv) > 1 else "locomo"
    suite = load_real_suite(name)
    total_msgs = sum(len(c.messages) for c in suite)
    total_qa = sum(len(c.facts) for c in suite)
    cats = Counter(f.category for c in suite for f in c.facts)
    abst = sum(1 for c in suite for f in c.facts if f.is_abstention)
    print(f"{name}: {len(suite)} cases, {total_msgs} messages, {total_qa} QA")
    print(f"  categories: {dict(cats)}")
    print(f"  abstention QA: {abst}")
    for c in suite[:3]:
        print(f"  {c.id}: {len(c.messages)} msgs, {len(c.facts)} qa")
