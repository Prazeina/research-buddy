# The Research Synthesizer

https://github.com/user-attachments/assets/015cf6d0-8163-4c06-a1d3-1eee773a3bbe

A local RAG tool for building a personal library of research papers and querying it with Ollama and ChromaDB. Your papers and your queries never leave your machine — useful for unpublished drafts, peer-review manuscripts, grant proposals, or any source material you can't send to cloud tools.

## 🧪 Quality Engineering: LLM Evaluation Harness & CI Gate

Beyond the RAG app itself, this repo ships an automated **evaluation harness** that tests the *quality* of the RAG's answers — the hard part of shipping LLM features, since the output is non-deterministic and can hallucinate, so `assert output == expected` is useless.

It uses [deepeval](https://github.com/confident-ai/deepeval) with an **LLM-as-judge** (Claude) to score every answer on four dimensions:

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved sources (not hallucinated)? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Contextual Precision** | Are the relevant retrieved chunks ranked above the noise? |
| **Contextual Recall** | Did retrieval fetch everything needed to answer? |

The metrics are **adversarially validated** — fabricated claims are injected to confirm the judge actually fails them rather than rubber-stamping every answer.

### CI Quality Gate

The eval runs as a **CI quality gate** (GitHub Actions — see [`.github/workflows/eval.yml`](.github/workflows/eval.yml)): every pull request to `main` is scored automatically, and the build **fails if average faithfulness regresses below threshold** — regression testing for non-deterministic LLM output. Because the RAG depends on local Ollama + ChromaDB, the gate runs on a self-hosted runner.

### Running the eval

```bash
# full eval — 4 metrics, saves a timestamped JSON report
python -m evals.faithfullness

# the CI gate check — faithfulness only, exits non-zero on regression
python -m evals.ci_gate
```

Requires the ChromaDB server + Ollama running, and `ANTHROPIC_API_KEY` set (the judge). The question set lives in [`evals/questions.json`](evals/questions.json), tagged by type (answerable / off-topic / false-premise / synthesis).

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
docker run -p 3001:3001 fengzhichao/chromadb-admin
```
*Note: the image was previously published as `flanker/chromadb-admin` but is now `fengzhichao/chromadb-admin`. Set the ChromaDB connection string inside the UI (see below) rather than via an env var.*

**Or using Node.js (Manual):**
```bash
git clone https://github.com/flanker/chromadb-admin.git
cd chromadb-admin
npm install
npm run dev
```
*Access the UI at `http://localhost:3001`.*

### Terminal 3: The Application
The CLI has three subcommands. Run `python3 rag.py --help` to see them.

**Add a paper to your library:**
```bash
python3 rag.py add path/to/paper.pdf
python3 rag.py add path/to/paper.pdf --title "Better Title If The Auto-Detect Picks A Bad One"
```
Re-adding the same PDF is a clean no-op (the paper_id is a content hash).

**List the papers you've ingested:**
```bash
python3 rag.py list
```

**Ask a question, across the whole library or scoped to one paper:**
```bash
python3 rag.py ask "what is the main contribution?"
python3 rag.py ask --paper <paper_id> "how does the method work?"
```
The `paper_id` for each paper appears in `rag list` — it's a short content hash of the PDF.

## The Web UI (easiest way to use it)

Prefer clicking to typing commands? There's a local web app with the same features
plus a **chat interface that remembers your conversation** (so follow-up questions
like "what are its limitations?" work).

You still need the ChromaDB server running (Terminal 1 above). Then, in another terminal:

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. From there you can:

- **Upload** a PDF straight from your browser to add it to the library
- **Browse and remove** papers in the sidebar, and scope questions to one paper
- **Chat** with your library — answers cite `(Title, p.N)`, and the conversation is
  saved to `chat_history.json` so it survives restarts (use **Clear conversation** to reset it)

Everything stays on your machine, exactly like the CLI. The chat history is local
and gitignored.

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
