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

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            embeddings.append(response['embedding'])
        return embeddings