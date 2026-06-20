"""CI quality gate: fail the build if AVERAGE faithfulness regresses below THRESHOLD.

Why average (not per-question pass/fail)? Some question types legitimately score
low on some metrics (e.g. an off-topic question has 0 contextual precision by
design), so a single per-question threshold would always fail. Gating on the
average faithfulness across the dataset is a stable, meaningful regression signal:
the build stays green at today's quality and goes red only if quality drops.

Exit code 0 = gate passed, 1 = gate failed (this is what CI keys off of).

Run locally:  python -m evals.ci_gate
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

# Baseline average faithfulness measured at ~0.54, so 0.45 leaves a small margin:
# green today, red only on a real regression. Raise this as the RAG improves.
THRESHOLD = 0.45


def main():
    client = core.connect_chroma()
    collection = core.get_library_collection(client)

    with open(os.path.join(os.path.dirname(__file__), "questions.json")) as f:
        questions = json.load(f)

    judge = AnthropicModel(model="claude-sonnet-4-6", max_tokens=4096)
    # Only faithfulness here — keeps the gate cheap (1 metric, not 4).
    metric = FaithfulnessMetric(threshold=0.7, model=judge, penalize_ambiguous_claims=True)

    test_cases = []
    for item in questions:
        result = core.answer_question(collection, item["question"])
        test_cases.append(LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            retrieval_context=result["retrieval_context"],
        ))

    res = evaluate(
        test_cases=test_cases,
        metrics=[metric],
        # low concurrency + throttle so we don't trip the API rate limit
        async_config=AsyncConfig(max_concurrent=2, throttle_value=1),
        display_config=DisplayConfig(print_results=False),
    )

    scores = [m.score for tr in res.test_results for m in tr.metrics_data]
    avg = sum(scores) / len(scores)
    print(f"Average faithfulness: {avg:.3f}  (gate threshold {THRESHOLD})")

    if avg < THRESHOLD:
        print("❌ QUALITY GATE FAILED — faithfulness regressed.")
        sys.exit(1)
    print("✅ QUALITY GATE PASSED.")


if __name__ == "__main__":
    main()
