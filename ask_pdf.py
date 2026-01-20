import ollama
import os
import argparse
import chromadb
import uuid
import re
from rag_utils import extract_text_from_pdf, chunk_text, OllamaEmbeddingFunction

def main():
    parser = argparse.ArgumentParser(description="Ask a question about a PDF using Ollama.")
    parser.add_argument("pdf_file", help="The path to the PDF file.")
    args = parser.parse_args()
    pdf_path = args.pdf_file

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return

    # Create a clean collection name from the filename
    # ChromaDB requires names to be alphanumeric, underscores, or hyphens
    filename = os.path.basename(pdf_path)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(filename)[0])
    collection_name = f"pdf_{clean_name}"[:63] # Limit length to 63 chars

    print("Connecting to ChromaDB Server (http://localhost:8000)...")
    try:
        # Connect to the running ChromaDB server (required for chromadb-admin)
        client = chromadb.HttpClient(host='localhost', port=8000)
        client.heartbeat()
    except Exception:
        print("\n❌ Error: Could not connect to ChromaDB server.")
        print("Please open a NEW terminal window and run:")
        print("chroma run --path ./chroma_db")
        return

    collection = client.get_or_create_collection(name=collection_name, embedding_function=OllamaEmbeddingFunction())

    # Only process the PDF if the collection is empty
    if collection.count() == 0:
        print("New PDF detected. Processing and embedding...")
        pdf_content = extract_text_from_pdf(pdf_path)
        if not pdf_content:
            print("Could not extract content from the PDF. Exiting.")
            return
        
        chunks = chunk_text(pdf_content)
        collection.add(
            documents=chunks,
            ids=[str(uuid.uuid4()) for _ in chunks]
        )
        print(f"Added {len(chunks)} chunks to the knowledge base.")
    else:
        print(f"Loaded existing knowledge base for '{filename}' ({collection.count()} chunks).")

    chat_history = []

    while True:
        question = input("\nWhat is your question about the PDF? (or type 'quit' to exit) ")
        if question.lower() == 'quit':
            break

        results = collection.query(query_texts=[question], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # Build history string (last 3 turns) to provide context without using too many tokens
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]])
        prompt = f"Based on the following context and conversation history, answer the question.\n\nContext:\n{context}\n\nHistory:\n{history_text}\n\nQuestion: {question}\n\nAnswer:"

        try:
            print("\nSending request to Ollama (model: llama3.2)...")
            response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}]
            )
            print("\nOllama's Response:")
            print(response['message']['content'])
            
            # Save the interaction to history
            chat_history.append((question, response['message']['content']))
        except Exception as e:
            print(f"\nAn error occurred while interacting with Ollama: {e}")
            print("Please ensure the Ollama server is running and the 'llama3.2' model is available.")

if __name__ == "__main__":
    main()