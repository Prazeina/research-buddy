import ollama
import pypdf
import os
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Ask a question about a PDF using Ollama.")
    parser.add_argument("pdf_file", help="The path to the PDF file.")
    args = parser.parse_args()
    pdf_path = args.pdf_file

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return

    pdf_content = extract_text_from_pdf(pdf_path)

    if not pdf_content:
        print("Could not extract content from the PDF. Exiting.")
        return

    print(f"Successfully extracted {len(pdf_content):,} characters from the PDF.")

    while True:
        question = input("\nWhat is your question about the PDF? (or type 'quit' to exit) ")
        if question.lower() == 'quit':
            break

        prompt = f"Based on the following document, answer the question.\n\nDocument:\n{pdf_content}\n\nQuestion: {question}\n\nAnswer:"

        try:
            print("\nSending request to Ollama (model: llama3.2)...")
            response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}]
            )
            print("\nOllama's Response:")
            print(response['message']['content'])
        except Exception as e:
            print(f"\nAn error occurred while interacting with Ollama: {e}")
            print("Please ensure the Ollama server is running and the 'llama3.2' model is available.")

if __name__ == "__main__":
    main()