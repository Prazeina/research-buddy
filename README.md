# The Research Synthesizer

A local RAG (Retrieval-Augmented Generation) tool that lets you chat with your PDF documents using Ollama and ChromaDB.

## Prerequisites

- **Python 3.10+**
- **Ollama** (installed and running)
- **Node.js** or **Docker** (for the Admin UI)

## Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Ollama Models
You need both the chat model and the embedding model:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## How to Run (The 3-Terminal Setup)

To use the full system with the Admin UI, you need three terminal windows running simultaneously.

### Terminal 1: ChromaDB Server
Start the database backend. This exposes your local data to the Admin UI.
```bash
chroma run --path ./chroma_db
```
*Note: It will listen on `localhost:8000`. You might see a 404 in the browser, which is normal.*

### Terminal 2: ChromaDB Admin UI
Run the visual dashboard to inspect your vectors.

**Using Docker:**
```bash
docker run -p 3001:3000 -e CHROMA_URL=http://host.docker.internal:8000 flanker/chromadb-admin
```

**Or using Node.js (Manual):**
```bash
git clone https://github.com/flanker/chromadb-admin.git
cd chromadb-admin
npm install
npm run dev
```
*Access the UI at `http://localhost:3001`.*

### Terminal 3: The Application
Run the tool to process PDFs and chat.

**Option A: Command Line Interface**
```bash
python3 ask_pdf.py your_file.pdf
```

**Option B: Web Interface (Streamlit)**
```bash
streamlit run app.py
```

## Configuring ChromaDB Admin

When you open `http://localhost:3001`, use these settings to connect:

- **Connection String:** `http://localhost:8000`
- **Tenant:** `default_tenant`
- **Database:** `default_database`
- **Embedding Model URL:** `http://localhost:11434/api/embeddings`
- **Embedding Model:** `nomic-embed-text`
- **Auth:** `No Auth`

## Acknowledgements

- **ChromaDB Admin**: Special thanks to flanker for the chromadb-admin UI which makes debugging vectors much easier.
- **Ollama**: For running local LLMs effortlessly.
