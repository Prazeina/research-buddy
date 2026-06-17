import os
import json
import hashlib
from datetime import datetime, timezone

import pypdf
import ollama
import chromadb


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_pages(pdf_path: str) -> list[str]:
    """Return the text of each page as a separate string (1-indexed by position).

    Raises FileNotFoundError or pypdf errors — callers decide how to surface them.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        return [page.extract_text() or "" for page in reader.pages]


def chunk_pages(pages: list[str], chunk_size: int = 1000, overlap: int = 100) -> list[tuple[int, str]]:
    """Chunk each page independently. Returns [(page_num, chunk_text), ...] with 1-indexed pages.

    A chunk never spans two pages, so its page number is unambiguous.
    """
    out: list[tuple[int, str]] = []
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            out.append((page_num, chunk))
    return out


def compute_paper_id(pdf_path: str) -> str:
    """Content-derived ID for a PDF: first 8 hex chars of sha1(file_bytes).

    Same bytes always produce the same ID — re-ingesting an identical file
    is a no-op rather than a duplicate. Reads in 64KB chunks so large PDFs
    don't blow up memory.
    """
    h = hashlib.sha1()
    with open(pdf_path, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()[:8]


def load_registry(path: str = 'papers.json') -> dict:
    """Load the paper registry from disk. Returns {} if the file doesn't exist yet."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_registry(registry: dict, path: str = 'papers.json') -> None:
    """Persist the registry. Sorted keys + indent so it diffs cleanly in git."""
    with open(path, 'w') as f:
        json.dump(registry, f, indent=2, sort_keys=True)


def register_paper(registry: dict, paper_id: str, filename: str, title: str) -> bool:
    """Add a paper to the registry in place. Returns True if added, False if already present.

    Caller is responsible for calling save_registry() after a successful add.
    """
    if paper_id in registry:
        return False
    registry[paper_id] = {
        'filename': filename,
        'title': title,
        'added_at': datetime.now(timezone.utc).isoformat(),
    }
    return True


def missing_ollama_models(required: list[str]) -> list[str]:
    """Return the subset of `required` model names not installed in Ollama.

    Tag suffixes (e.g. ':latest') are ignored when comparing — `llama3.2`
    matches an installed `llama3.2:latest`. If Ollama itself is unreachable,
    all required models are treated as missing.
    """
    try:
        installed = {m.model.split(':')[0] for m in ollama.list().models}
    except Exception:
        return list(required)
    return [name for name in required if name not in installed]


def embed_query(text: str, model: str = 'nomic-embed-text') -> list[float]:
    """Embed a search query with nomic-embed-text's 'search_query:' task prefix.

    nomic-embed-text was trained to expect a task prefix on every input. Queries
    and documents use *different* prefixes, so they can't share one code path:
    documents go through OllamaEmbeddingFunction ('search_document:'), and queries
    come through here ('search_query:'). Using the right prefix on each side is
    what makes question<->passage matching work as the model intends.
    """
    response = ollama.embeddings(model=model, prompt=f"search_query: {text}")
    return response['embedding']


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Embeds *documents* via Ollama, with nomic-embed-text's 'search_document:' prefix.

    Chroma calls this automatically for collection.add(), so this is the document
    side of the prefix pair. Queries are embedded separately via embed_query() and
    passed in as query_embeddings — see core.retrieve().
    """

    def __init__(self, model: str = 'nomic-embed-text', prefix: str = 'search_document: '):
        self.model = model
        self.prefix = prefix

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model, prompt=f"{self.prefix}{text}")
            embeddings.append(response['embedding'])
        return embeddings
