"""B6 — pytest-style regression suite for the RAG.

Each question in questions.json becomes a separate test. `assert_test` runs all
metrics and FAILS the test if any metric scores below its threshold — exactly
like a normal unit test, but for LLM answer quality. This is what makes the eval
a CI-ready quality gate.

Run it with deepeval's own pytest runner (Chroma + Ollama up, ANTHROPIC_API_KEY set):
    deepeval test run evals/test_rag_eval.py
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import core

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import AnthropicModel

# --- one-time setup (runs once when pytest collects this file) ---
_client = core.connect_chroma()
_collection = core.get_library_collection(_client)

with open(os.path.join(os.path.dirname(__file__), "questions.json")) as f:
    QUESTIONS = json.load(f)

_judge = AnthropicModel(model="claude-sonnet-4-6", max_tokens=4096)
METRICS = [
    FaithfulnessMetric(threshold=0.7, model=_judge, penalize_ambiguous_claims=True),
    AnswerRelevancyMetric(threshold=0.7, model=_judge),
    ContextualPrecisionMetric(threshold=0.7, model=_judge),
    ContextualRecallMetric(threshold=0.7, model=_judge),
]


# one test per question; the `ids` give each test a readable name (its type tag)
@pytest.mark.parametrize("item", QUESTIONS, ids=[q["type"] for q in QUESTIONS])
def test_rag_answer(item):
    # ask the RAG
    result = core.answer_question(_collection, item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected"],
        retrieval_context=result["retrieval_context"],
    )
    # fails the test if ANY metric is below its threshold
    assert_test(test_case, METRICS)
