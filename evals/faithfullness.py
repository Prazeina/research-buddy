"""Step 3: score one RAG answer for FAITHFULNESS using deepeval, judged by Claude.

Faithfulness = what fraction of the answer's claims are actually supported by the
retrieved context. The judge (Claude) breaks the answer into claims, checks each
against retrieval_context, and returns score = supported / total.

Run from the repo root (venv active, Chroma + Ollama up, ANTHROPIC_API_KEY set):
    python -m evals.faithfullness
"""

# --- make `import core` work no matter how this file is launched ---
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import AnthropicModel


def main():
    # 1. Get one answer from the RAG (same three calls as the Runner).
    client = core.connect_chroma()
    collection = core.get_library_collection(client)

    question = "What is the main contribution of this paper?"
    result = core.answer_question(collection, question)

    # 2. Package the three pieces deepeval needs into a test case.
    #    input -> the question, actual_output -> the answer,
    #    retrieval_context -> the chunks the answer was built from.
    test_case = LLMTestCase(
        input=question,
        # actual_output=result["answer"],
        actual_output = "Contiformer was invented by Albert Einstein in 1905 and runs on quantum computers.",
        retrieval_context=result["retrieval_context"],
    )

    # 3. Set up the judge (Claude) and the metric.
    #    threshold=0.7 -> "pass" if at least 70% of claims are grounded.
    judge = AnthropicModel(model="claude-sonnet-4-6")
    metric = FaithfulnessMetric(threshold=0.7, model=judge, penalize_ambiguous_claims=True)

    # 4. Run the judge and read the result.
    metric.measure(test_case)

    print("QUESTION    :", question)
    # Print what is ACTUALLY being judged (test_case.actual_output), not result["answer"],
    # so the output never lies about what the metric scored.
    print("JUDGED OUTPUT:", (test_case.actual_output or "")[:300], "...")
    print("-" * 60)
    print(f"FAITHFULNESS: {metric.score}  (threshold {metric.threshold}) "
          f"-> {'PASS' if metric.is_successful() else 'FAIL'}")
    print("REASON      :", metric.reason)


if __name__ == "__main__":
    main()
