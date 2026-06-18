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
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import AnthropicModel


def main():
    # 1. Get one answer from the RAG (same three calls as the Runner).
    client = core.connect_chroma()
    collection = core.get_library_collection(client)

    questions = ["What is the main contribution of this paper?", 
                "What model architecture does the paper propose", 
                "What is the capital of Nepal?",
                "Is the dataset publicly available?",
                "In the future work section it discussed about flying cars?",
                "What are the main limitations of the proposed method?"]

    # 2. Set up the judge (Claude) and the metric.
    judge = AnthropicModel(model="claude-haiku-4-5", max_tokens=4096)
    metric = FaithfulnessMetric(threshold=0.7, model=judge, penalize_ambiguous_claims=True)
    
    # 3. Package the three pieces deepeval needs into a test case.
    #    input -> the question, actual_output -> the answer,
    #    retrieval_context -> the chunks the answer was built from.

    results = []  # collect every question's result so we can save them at the end

    for question in questions:
        result = core.answer_question(collection, question)
        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=result["retrieval_context"],
        )
        # 4. Run the judge and read the result.
        metric.measure(test_case)
        verdict = "PASS" if metric.is_successful() else "FAIL"
        print("=" * 70)
        print(f"Q: {question}")
        print(f"SCORE: {round(metric.score, 2)} (threshold {metric.threshold}) -> {verdict}")
        print(f"WHY  : {metric.reason}")

        # keep a structured record of this question's result
        results.append({
            "question": question,
            "answer": result["answer"],
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": metric.reason,
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
