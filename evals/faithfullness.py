"""Step 3: score one RAG answer for FAITHFULNESS using deepeval, judged by Claude.

Faithfulness = what fraction of the answer's claims are actually supported by the
retrieved context. The judge (Claude) breaks the answer into claims, checks each
against retrieval_context, and returns score = supported / total.

Run from the repo root (venv active, Chroma + Ollama up, ANTHROPIC_API_KEY set):
    python -m evals.faithfullness
"""

# --- make `import core` work no matter how this file is launched ---
import json
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import AnthropicModel


def main():
    # Connect to ChromaDB
    client = core.connect_chroma()
    collection = core.get_library_collection(client)

    questions_path = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(questions_path) as f:
        questions_data = json.load(f)

    # 2. Set up the judge (Claude) and the metric.
    judge = AnthropicModel(model="claude-haiku-4-5", max_tokens=4096)
    metric = FaithfulnessMetric(threshold=0.7, model=judge, penalize_ambiguous_claims=True)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
    
    # 3. Package the three pieces deepeval needs into a test case. input -> the question, actual_output -> the answer, retrieval_context -> the chunks the answer was built from.

    print(f"Loaded {len(questions_data)} questions from questions.json")

    results = []  # collect every question's result so we can save them at the end

    for item in questions_data:
        question = item["question"]   # the question text
        qtype = item["type"]          # the tag (answerable / off-topic / ...)
        result = core.answer_question(collection, question)
        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=result["retrieval_context"],
        )
        # 4. Run the judge and read the result.
        metric.measure(test_case)
        relevancy.measure(test_case)
        verdict = "PASS" if metric.is_successful() else "FAIL"

        # keep a structured record of this question's result
        results.append({
            "question": question,
            "type": qtype,
            "answer": result["answer"],
            "faithfullness_score": metric.score,
            "faithfullness_passed": metric.is_successful(),
            "faithfullness_reason": metric.reason,
            "relevancy_score": relevancy.score,
            "relevancy_passed": metric.is_successful(),
            "relevancy_reason": relevancy.reason
        })

    # 5. Save all results to a timestamped JSON file so no run overwrites another.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(__file__), f"results_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 70)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
