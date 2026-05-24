import os
import pypdf
import ollama
import chromadb

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return ""

    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

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


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model: str = 'nomic-embed-text'):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings