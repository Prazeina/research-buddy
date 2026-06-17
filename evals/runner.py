"""Walking-skeleton Runner for the eval harness.

Step 2 of the build: prove we can drive the RAG from a separate script and
SEE everything it returns — especially the retrieval_context. No scoring yet.

Run it from the repo root (with the venv active and the Chroma server running):
    python -m evals.runner
"""

# --- make `import core` work no matter how this file is launched ---
# core.py lives in the repo root (one level up from evals/), so we add that
# folder to Python's import path before importing it.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core


def main():
    # 1. Connect to the RAG's vector store (same calls the CLI/web app use).
    client = core.connect_chroma()
    collection = core.get_library_collection(client)

    # 2. Ask the RAG one question.
    question = "What is the main contribution of this paper?"
    result = core.answer_question(collection, question)

    # 3. Look at everything that came back.
    print("QUESTION       :", question)
    print("SEARCH QUERY   :", result["search_query"])
    print("ANSWER         :", result["answer"])

    # The piece we worked to expose: the chunks the answer was built from.
    context = result["retrieval_context"]
    print(f"RETRIEVAL CTX  : {len(context)} chunk(s) retrieved")
    if context:
        print("  chunk[0] preview:", context[0][:200].replace("\n", " "), "...")


if __name__ == "__main__":
    main()

