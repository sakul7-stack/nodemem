"""
datasets.py -- deterministic synthetic conversation suite.

Each case is a multi-turn conversation across one of the paper's five domains,
with:
  * planted *critical facts* (entity/number/date) and a QA probe for each, and
  * a subset of *fact updates* (a later turn overrides an earlier value), which
    is the adversarial case for append-only notes.

The optional ``long`` scale extends the same deterministic generator to 120,
240, and 480-message stress cases for token-growth experiments.

Fully reproducible from a seed -- no API needed to build the data, so the
suite is identical on every machine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Fact:
    key: str
    question: str
    answer: str          # the CURRENT correct answer (post-update if updated)
    updated: bool = False
    category: str = ""           # benchmark question category (real datasets)
    is_abstention: bool = False  # gold answer is "not answerable" (abstention probe)
    evidence: list = field(default_factory=list)  # gold evidence ids (optional)


@dataclass
class Case:
    id: str
    domain: str
    length_class: str
    messages: list[dict] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)


# Domain -> (entity pool, filler user lines, filler assistant lines)
_DOMAINS = {
    "product_development": {
        "entities": ["Project Atlas", "Acme Corp", "the Rust backend", "Postgres", "v2.3"],
        "fillers": [
            "Let's keep iterating on the roadmap.",
            "Can you summarize where we are on the migration?",
            "I think the API surface needs another pass.",
            "What about the onboarding flow for new users?",
            "We should revisit the caching layer next sprint.",
        ],
    },
    "research_planning": {
        "entities": ["the LoCoMo benchmark", "Dr. Patel", "the ablation study", "Section 4", "NeurIPS"],
        "fillers": [
            "How should we structure the experiments?",
            "Let's outline the related work section.",
            "I want the baselines to be reproducible.",
            "Which datasets give the strongest signal?",
            "We need to nail down the evaluation protocol.",
        ],
    },
    "travel_booking": {
        "entities": ["Lufthansa", "the Marriott downtown", "Terminal 2", "the aisle seat", "Frankfurt"],
        "fillers": [
            "I'd like to finalize the itinerary.",
            "Are there earlier connections available?",
            "Please double-check the baggage allowance.",
            "What's the cancellation policy again?",
            "Can we add a day to the trip?",
        ],
    },
    "technical_debugging": {
        "entities": ["the auth service", "Redis", "the 502 errors", "the staging cluster", "pod 7"],
        "fillers": [
            "The latency spikes are back.",
            "Walk me through the stack trace.",
            "Did the last deploy change the config?",
            "Let's check the connection pool settings.",
            "Could it be a memory leak?",
        ],
    },
    "business_strategy": {
        "entities": ["the EMEA expansion", "Q3", "the enterprise tier", "Northwind Ltd", "the channel partners"],
        "fillers": [
            "Where do we stand on the GTM plan?",
            "What's our pricing hypothesis?",
            "Let's align on the hiring plan.",
            "How are the partner negotiations going?",
            "We should model the unit economics.",
        ],
    },
}

# Fact templates: (key, statement_fmt, question, value_pool)
_FACT_TEMPLATES = [
    ("budget", "The approved budget for {ent} is ${val}.",
     "What is the approved budget for {ent}?", ["50,000", "120,000", "85,000", "240,000"]),
    ("deadline", "The hard deadline for {ent} is {val}.",
     "What is the hard deadline for {ent}?", ["March 14", "September 2", "June 30", "November 9"]),
    ("owner", "The owner of {ent} is {val}.",
     "Who owns {ent}?", ["Priya", "Marcus", "Elena", "Tomas"]),
    ("metric", "The target metric for {ent} is {val} percent.",
     "What is the target metric for {ent}?", ["92", "78", "65", "99"]),
    ("constraint", "A hard constraint on {ent} is that it must {val}.",
     "What hard constraint applies to {ent}?",
     ["stay under 200ms p95", "support offline mode", "be GDPR compliant", "run on a single node"]),
]


def _gen_case(rng: random.Random, idx: int, domain: str, length_class: str,
              n_messages: int, n_facts: int, n_updates: int) -> Case:
    pool = _DOMAINS[domain]
    entities = pool["entities"]
    fillers = pool["fillers"]

    templates = rng.sample(_FACT_TEMPLATES, k=min(n_facts, len(_FACT_TEMPLATES)))
    facts: list[Fact] = []
    fact_statements: list[str] = []   # assistant lines that carry a fact
    update_statements: list[tuple[int, str]] = []  # (insert position, line)

    for ti, (key, stmt_fmt, q_fmt, values) in enumerate(templates):
        ent = entities[ti % len(entities)]
        val = rng.choice(values)
        fact_statements.append(stmt_fmt.format(ent=ent, val=val))
        answer = val
        updated = False
        if ti < n_updates:
            # later override with a different value
            new_val = rng.choice([v for v in values if v != val])
            update_statements.append((ti, stmt_fmt.format(ent=ent, val=new_val)
                                      .replace("is ", "has been updated to ", 1)))
            answer = new_val
            updated = True
        facts.append(Fact(key=f"{key}:{ent}",
                          question=q_fmt.format(ent=ent),
                          answer=answer, updated=updated))

    # Build the message list: interleave fillers, fact statements, and updates.
    messages: list[dict] = []
    # opening
    messages.append({"role": "user", "content":
                     f"Hi, I want to work through some details on {entities[0]}."})
    messages.append({"role": "assistant", "content":
                     "Sure, let's go through it. I'll keep track of the key details."})

    # place facts early-to-mid
    fact_iter = iter(fact_statements)
    update_iter = iter(s for _, s in update_statements)

    target_turns = max(n_messages, 2 + 2 * (len(fact_statements) + len(update_statements)))
    placed_facts = 0
    placed_updates = 0
    while len(messages) < target_turns:
        # user filler
        messages.append({"role": "user", "content": rng.choice(fillers)})
        # assistant turn: sometimes carry a fact, then an update later
        progress = len(messages) / target_turns
        if placed_facts < len(fact_statements) and progress < 0.6:
            messages.append({"role": "assistant", "content":
                             "Noted. " + next(fact_iter)})
            placed_facts += 1
        elif placed_updates < len(update_statements) and progress > 0.55:
            messages.append({"role": "assistant", "content":
                             "Quick correction: " + next(update_iter)})
            placed_updates += 1
        else:
            messages.append({"role": "assistant", "content":
                             "Understood. Let me note that and continue."})

    # flush any unplaced facts/updates near the end so they always appear
    for s in fact_iter:
        messages.append({"role": "assistant", "content": "Also: " + s})
    for s in update_iter:
        messages.append({"role": "assistant", "content": "Correction: " + s})

    return Case(id=f"{domain}-{length_class}-{idx}", domain=domain,
                length_class=length_class, messages=messages, facts=facts)


def build_suite(seed: int = 7, scale: str = "small") -> list[Case]:
    """Build the conversation suite.

    scale='small' -> 8 cases (cheap, for a real-API run).
    scale='paper' -> 20 cases matching the paper's stated counts.
    scale='long' -> 12 stress cases with >100 messages for token-growth curves.
    """
    rng = random.Random(seed)
    cases: list[Case] = []

    if scale == "paper":
        plan = [("short", 7, 14, 3, 1), ("medium", 7, 32, 4, 2), ("long", 6, 60, 5, 2)]
    elif scale == "long":
        plan = [("long120", 4, 120, 5, 2), ("long240", 4, 240, 5, 2), ("long480", 4, 480, 5, 2)]
    elif scale == "tiny":
        plan = [("short", 2, 14, 3, 1), ("medium", 2, 30, 4, 2), ("long", 2, 56, 5, 2)]
    else:  # small
        plan = [("short", 3, 14, 3, 1), ("medium", 3, 30, 4, 2), ("long", 2, 56, 5, 2)]

    domains = list(_DOMAINS.keys())
    counter = 0
    for length_class, n_cases, n_msgs, n_facts, n_updates in plan:
        for i in range(n_cases):
            domain = domains[counter % len(domains)]
            cases.append(_gen_case(rng, i, domain, length_class, n_msgs, n_facts, n_updates))
            counter += 1
    return cases


if __name__ == "__main__":
    suite = build_suite(scale="small")
    for c in suite:
        print(f"{c.id}: {len(c.messages)} msgs, {len(c.facts)} facts "
              f"({sum(f.updated for f in c.facts)} updated)")
