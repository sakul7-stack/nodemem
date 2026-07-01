"""
judge.py -- LLM-as-judge scoring for QA correctness, replacing the crude
token-containment match used in the synthetic pilot.

The grading prompts are taken from the official LongMemEval evaluation protocol
(Wu et al., ICLR 2025, ``src/evaluation/evaluate_qa.py``) so our numbers are
comparable to that benchmark's methodology. The only change is that the judge
can be served by any OpenAI-compatible chat model (here, a Groq-hosted Llama),
not just GPT-4o, which keeps replication free. A containment fallback is kept
for offline plumbing tests.

Each internal category name is mapped to the matching LongMemEval "task" so the
right rubric (default / temporal off-by-one / knowledge-update / preference /
abstention) is applied.
"""

from __future__ import annotations

import re

# Internal category -> LongMemEval task name used to pick the grading rubric.
CATEGORY_TO_TASK = {
    # LoCoMo
    "single_hop": "single-session-user",
    "multi_hop": "multi-session",
    "open_domain": "multi-session",
    "temporal": "temporal-reasoning",
    "adversarial": "abstention",
    # LongMemEval
    "info_extraction": "single-session-user",
    "multi_session": "multi-session",
    "knowledge_update": "knowledge-update",
    "preference": "single-session-preference",
    "abstention": "abstention",
}

_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no.\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: "
    "{r}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, etc., "
    "and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct.\n\nQuestion: {q}\n\nCorrect "
    "Answer: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes "
    "or no only."
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {q}\n\nCorrect Answer: "
    "{a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
)

_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the "
    "points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly.\n\nQuestion: {q}\n\nRubric: "
    "{a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\nQuestion: "
    "{q}\n\nExplanation: {a}\n\nModel Response: {r}\n\nDoes the model correctly "
    "identify the question as unanswerable? Answer yes or no only."
)

_TASK_TEMPLATES = {
    "single-session-user": _DEFAULT,
    "single-session-assistant": _DEFAULT,
    "multi-session": _DEFAULT,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
    "abstention": _ABSTENTION,
}

_ABSTAIN_PHRASES = (
    "no information", "not mentioned", "not available", "cannot answer",
    "can't answer", "don't know", "do not know", "unanswerable", "no mention",
    "isn't mentioned", "is not mentioned", "not specified", "not provided",
    "not stated", "unable to", "insufficient information", "not enough information",
)


def build_judge_prompt(category: str, question: str, gold: str, response: str,
                       is_abstention: bool) -> str:
    task = "abstention" if is_abstention else CATEGORY_TO_TASK.get(category, "multi-session")
    template = _TASK_TEMPLATES.get(task, _DEFAULT)
    return template.format(q=question, a=gold, r=response)


def containment_correct(response: str, gold: str, is_abstention: bool) -> bool:
    """Cheap offline fallback. For abstention, reward declining to answer."""
    out = re.sub(r"[^a-z0-9 ]", " ", response.lower())
    out = re.sub(r"\s+", " ", out).strip()
    if is_abstention:
        return any(p in response.lower() for p in _ABSTAIN_PHRASES)
    g = re.sub(r"[^a-z0-9 ]", " ", gold.lower())
    g = re.sub(r"\s+", " ", g).strip()
    return bool(g) and all(tok in out for tok in g.split())


def judge_answer(client, model: str, category: str, question: str, gold: str,
                 response: str, is_abstention: bool = False) -> bool:
    """Grade one (question, gold, response) triple with the LLM judge.

    Falls back to containment if the judge call fails so a single transient
    error never aborts a long run.
    """
    # Fast path: an explicit abstention is unambiguous and saves a judge call.
    if is_abstention and containment_correct(response, gold, True):
        return True
    prompt = build_judge_prompt(category, question, gold, response, is_abstention)
    try:
        c = client.complete(model, None, [{"role": "user", "content": prompt}],
                            max_tokens=10)
        return "yes" in c.text.strip().lower()[:12]
    except Exception:
        return containment_correct(response, gold, is_abstention)
