#!/usr/bin/env python3
"""judge_agreement.py -- cross-validate the LLM judge with a second model family.

For a sample of LoCoMo questions we generate one real answer (synthesizer over a
BM25-retrieval context) and grade it with two judges from different model
families. We report observed agreement and Cohen's kappa, so the headline
numbers are not hostage to a single judge.

Primary judge defaults to llama-3.3-70b-versatile; the second family defaults to
openai/gpt-oss-120b (both on Groq). Usage:
  python scripts/judge_agreement.py [n_questions]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import load_dotenv_if_present, env_keys           # noqa: E402
from real_datasets import load_locomo                            # noqa: E402
from hierarchical_memory import OpenAICompatClient, build_retriever, RetrievalRAG  # noqa: E402
from judge import judge_answer                                   # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SYNTH = "meta-llama/llama-4-scout-17b-16e-instruct"
JUDGE_A = "llama-3.3-70b-versatile"
JUDGE_B = "openai/gpt-oss-120b"


def sample_questions(n: int):
    """A category-balanced sample across conversations."""
    suite = load_locomo(max_qa_per_conv=8, seed=11)
    rows = []
    for case in suite:
        for fact in case.facts:
            rows.append((case, fact))
    # round-robin by category for balance
    by = {}
    for case, fact in rows:
        by.setdefault(fact.category, []).append((case, fact))
    out, i = [], 0
    cats = sorted(by)
    while len(out) < n and any(i < len(by[c]) for c in cats):
        for c in cats:
            if i < len(by[c]):
                out.append(by[c][i])
                if len(out) >= n:
                    break
        i += 1
    return out


def main():
    load_dotenv_if_present()
    keys = env_keys("GROQ_API_KEY", "GROQ_API_KEYS")
    client = OpenAICompatClient(api_keys=keys, base_url="https://api.groq.com/openai/v1")

    samples = sample_questions(N)
    agree = 0
    a_yes = b_yes = both_yes = 0
    n = 0
    print(f"Generating + dual-judging {len(samples)} answers "
          f"(synth={SYNTH}, judgeA={JUDGE_A}, judgeB={JUDGE_B})...")
    for case, fact in samples:
        retr = build_retriever(case, backend="bm25")
        system, msgs = RetrievalRAG(6, retr, top_k=5).build_context(case.messages, fact.question)
        try:
            ans = client.complete(SYNTH, system, msgs, max_tokens=160).text
        except Exception as e:
            print("  (skip, gen failed:", str(e)[:50], ")")
            continue
        la = judge_answer(client, JUDGE_A, fact.category, fact.question, fact.answer, ans, fact.is_abstention)
        lb = judge_answer(client, JUDGE_B, fact.category, fact.question, fact.answer, ans, fact.is_abstention)
        n += 1
        agree += int(la == lb)
        a_yes += int(la); b_yes += int(lb); both_yes += int(la and lb)

    po = agree / n if n else 0.0
    pa, pb = a_yes / n, b_yes / n
    pe = pa * pb + (1 - pa) * (1 - pb)   # expected agreement under independence
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    print(f"\nJudge cross-validation (n={n}):")
    print(f"  judge A ({JUDGE_A}) accuracy on sample: {pa:.1%}")
    print(f"  judge B ({JUDGE_B}) accuracy on sample: {pb:.1%}")
    print(f"  observed agreement: {po:.1%}")
    print(f"  Cohen's kappa:      {kappa:.3f}")


if __name__ == "__main__":
    main()
