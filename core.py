"""Shared library logic used by BOTH the CLI (rag.py) and the web UI (app.py).

Everything here RETURNS data (and raises on failure) instead of printing or calling
sys.exit. That's the whole point: a UI can't use print()/exit(), so the reusable
logic lives here and the CLI wraps it with its own friendly messages.
"""
import os

import chromadb
import ollama

from rag_utils import (
    OllamaEmbeddingFunction,
    chunk_pages,
    compute_paper_id,
    embed_query,
    extract_pages,
    register_paper,
)

LIBRARY_COLLECTION = 'library'
CHAT_MODEL = 'llama3.2'

# How many prior conversation turns to feed the model. Persisted history can grow
# without bound, but the model has a limited context window, so we only replay the
# most recent turns (see app.py / answer_question).
HISTORY_WINDOW = 6


def connect_chroma(host: str = 'localhost', port: int = 8000) -> chromadb.api.ClientAPI:
    """Return a live Chroma client. Raises if the server isn't reachable."""
    client = chromadb.HttpClient(host=host, port=port)
    client.heartbeat()
    return client


def get_library_collection(client):
    """Open (or create) the shared library collection.

    New collections use cosine distance — the right metric for text embeddings,
    where direction matters and magnitude shouldn't. (Chroma's default is L2.)
    Note: this only applies when the collection is first created; an existing
    collection keeps whatever space it was born with.
    """
    return client.get_or_create_collection(
        name=LIBRARY_COLLECTION,
        embedding_function=OllamaEmbeddingFunction(),
        metadata={'hnsw:space': 'cosine'},
    )


def detect_title(pages: list[str], fallback: str) -> str:
    """First non-empty line of page 1, capped to 100 chars. Falls back to filename stem."""
    if pages:
        for line in pages[0].splitlines():
            line = line.strip()
            if line:
                return line[:100]
    return fallback


def ingest_pdf(pdf_path: str, collection, registry: dict, title_override: str | None = None) -> dict:
    """Ingest one PDF into the library. Returns a result dict; never prints.

    The returned dict always has a 'status' of 'added' | 'skipped' | 'failed':
      added   -> paper_id, title, filename, n_chunks, n_pages
      skipped -> paper_id, title, filename   (identical bytes already ingested)
      failed  -> filename, error

    Mutates `registry` in place on success; the caller is responsible for
    persisting it with save_registry().
    """
    basename = os.path.basename(pdf_path)
    paper_id = compute_paper_id(pdf_path)

    if paper_id in registry:
        return {
            'status': 'skipped',
            'paper_id': paper_id,
            'title': registry[paper_id]['title'],
            'filename': basename,
        }

    try:
        pages = extract_pages(pdf_path)
    except Exception as e:
        return {'status': 'failed', 'filename': basename, 'error': f"Could not read PDF: {e}"}

    title = title_override or detect_title(pages, os.path.splitext(basename)[0])
    chunks = chunk_pages(pages)

    if not chunks:
        return {
            'status': 'failed',
            'filename': basename,
            'error': 'No text could be extracted (image-only scan?). Run OCR on it first.',
        }

    chunk_ids = [f"{paper_id}-{i}" for i in range(len(chunks))]
    try:
        collection.add(
            documents=[text for _, text in chunks],
            metadatas=[
                {'paper_id': paper_id, 'title': title, 'page': page, 'chunk_type': 'paper'}
                for page, text in chunks
            ],
            ids=chunk_ids,
        )
    except Exception as e:
        # Roll back any partial write so a failed add never leaves orphan chunks.
        try:
            collection.delete(ids=chunk_ids)
        except Exception:
            pass
        return {'status': 'failed', 'filename': basename, 'error': f"Indexing failed: {e}"}

    register_paper(registry, paper_id, filename=basename, title=title)
    return {
        'status': 'added',
        'paper_id': paper_id,
        'title': title,
        'filename': basename,
        'n_chunks': len(chunks),
        'n_pages': len(pages),
    }


def remove_paper(collection, registry: dict, paper_id: str) -> int:
    """Delete a paper's chunks from Chroma and drop its registry entry (in place).

    Returns the number of chunks deleted. Caller persists the registry afterwards.
    """
    existing = collection.get(where={'paper_id': paper_id}, include=[])
    count = len(existing['ids'])
    collection.delete(where={'paper_id': paper_id})
    registry.pop(paper_id, None)
    return count


def retrieve(collection, question: str, paper_id: str | None = None, top: int = 5):
    """Retrieve the top-k most relevant chunks. Returns (documents, metadatas).

    The query is embedded here (with the 'search_query:' prefix) and passed as
    query_embeddings, rather than letting Chroma embed it as a document — that's
    how the question gets the *query* prefix while stored chunks keep the
    *document* prefix.
    """
    query_kwargs = {'query_embeddings': [embed_query(question)], 'n_results': top}
    if paper_id:
        query_kwargs['where'] = {'paper_id': paper_id}
    results = collection.query(**query_kwargs)
    return results['documents'][0], results['metadatas'][0]


def condense_question(question: str, history: list | None) -> str:
    """Rewrite a follow-up into a standalone question using recent chat turns.

    Retrieval needs a self-contained query: "what about its limitations?" matches
    nothing on its own. With no history, the question is returned unchanged.
    """
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:])
    prompt = (
        "Given the conversation so far and a follow-up question, rewrite the follow-up "
        "as a standalone question that makes sense without the conversation. "
        "Reply with ONLY the rewritten question, nothing else.\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone question:"
    )
    resp = ollama.chat(model=CHAT_MODEL, messages=[{'role': 'user', 'content': prompt}])
    return resp['message']['content'].strip() or question


def build_prompt(question: str, documents: list[str], metadatas: list[dict]) -> str:
    """Assemble the grounded, citation-instructed prompt from retrieved sources."""
    context = "\n\n".join(
        f"[Source {i}: {meta['title']}, p.{meta['page']}]\n{doc}"
        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1)
    )
    return (
        "You are a research assistant helping a researcher synthesize material from their paper library. "
        "Answer the question using the sources below. Lead with a direct answer, not a disclaimer. "
        "The answer rarely appears as one explicit sentence, so synthesize and draw reasonable "
        "inferences across the relevant sources rather than demanding an exact match. "
        "After each claim, cite the source you used with (Title, p.N) where N is the page number. "
        "Do not rely on knowledge beyond the sources. "
        "Only if the sources are genuinely unrelated to the question should you say the library "
        "doesn't cover it; otherwise give your best grounded answer.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def answer_question(
    collection,
    question: str,
    paper_id: str | None = None,
    top: int = 5,
    history: list | None = None,
) -> dict:
    """Run one full RAG turn. Returns {answer, sources, search_query, retrieval_context}.

    `history` is an optional list of {'role', 'content'} dicts for conversational
    follow-ups. When present, the follow-up is first condensed into a standalone
    query for retrieval, and recent turns are replayed to the model for context.
    `answer` is None when nothing relevant was found.
    """
    search_query = condense_question(question, history) if history else question
    documents, metadatas = retrieve(collection, search_query, paper_id, top)

    if not documents:
        return {'answer': None, 'sources': [], 'search_query': search_query, 'retrieval_context': []}

    messages = []
    if history:
        messages.extend(
            {'role': m['role'], 'content': m['content']} for m in history[-HISTORY_WINDOW:]
        )
    messages.append({'role': 'user', 'content': build_prompt(question, documents, metadatas)})

    resp = ollama.chat(model=CHAT_MODEL, messages=messages)
    sources = [
        {'title': m['title'], 'page': m['page'], 'paper_id': m['paper_id']}
        for m in metadatas
    ]
    return {'answer': resp['message']['content'], 'sources': sources, 'search_query': search_query, 'retrieval_context': documents}
