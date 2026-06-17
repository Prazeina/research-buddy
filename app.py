"""The Research Synthesizer -- local web UI (Streamlit).

Run it with:   streamlit run app.py
It opens in your browser at http://localhost:8501. Like the CLI, everything stays
on your machine: the PDFs, the vector DB, and your saved conversation.

Needs the ChromaDB server running in another terminal:
    chroma run --path ./chroma_db
"""
import os
import tempfile

import streamlit as st

import core
from chat_store import load_chat, save_chat, clear_chat
from rag_utils import load_registry, save_registry

REGISTRY_PATH = 'papers.json'

st.set_page_config(page_title="The Research Synthesizer", page_icon="📚", layout="wide")


@st.cache_resource
def get_collection():
    """Open the library once and reuse it across Streamlit reruns."""
    client = core.connect_chroma()
    return core.get_library_collection(client)


def open_library_or_stop():
    try:
        return get_collection()
    except Exception:
        st.error(
            "Can't reach ChromaDB at localhost:8000.\n\n"
            "Start it in another terminal:  `chroma run --path ./chroma_db`"
        )
        st.stop()


# Conversation lives in session_state during the run, but is loaded from / saved to
# disk so it survives restarts.
if 'messages' not in st.session_state:
    st.session_state.messages = load_chat()

collection = open_library_or_stop()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            st.caption(f"- {s['title']}, p.{s['page']}  ·  `{s['paper_id']}`")


# --------------------------------------------------------------------------- #
# Sidebar: the library (upload / list / remove / scope)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📚 Library")

    uploaded = st.file_uploader("Add a paper", type="pdf")
    if uploaded is not None and st.button("Add to library", use_container_width=True):
        with st.spinner(f"Reading and indexing {uploaded.name}..."):
            registry = load_registry(REGISTRY_PATH)
            # ingest_pdf needs a real file to hash + read, so spool the upload to disk.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                result = core.ingest_pdf(tmp_path, collection, registry)
            finally:
                os.unlink(tmp_path)

            if result['status'] == 'added':
                # The hash is content-based, but the stored filename was the temp
                # name — replace it with the real upload name.
                registry[result['paper_id']]['filename'] = uploaded.name
                save_registry(registry, REGISTRY_PATH)
                st.success(f"Added: {result['title']}  ({result['n_chunks']} chunks)")
            elif result['status'] == 'skipped':
                st.info(f"Already in library: {result['title']}")
            else:
                st.error(result['error'])
        st.rerun()

    st.divider()

    registry = load_registry(REGISTRY_PATH)
    papers = sorted(registry.items(), key=lambda kv: kv[1]['added_at'], reverse=True)

    scope_id = None
    if not papers:
        st.caption("No papers yet — upload one above to get started.")
    else:
        scope_options = {"Whole library": None}
        for pid, meta in papers:
            scope_options[f"{meta['title'][:40]}  ({pid})"] = pid
        scope_label = st.selectbox("Ask about:", list(scope_options.keys()))
        scope_id = scope_options[scope_label]

        st.subheader(f"{len(papers)} paper{'s' if len(papers) != 1 else ''}")
        for pid, meta in papers:
            row, btn = st.columns([5, 1])
            row.markdown(f"**{meta['title'][:48]}**  \n`{pid}`")
            if btn.button("🗑", key=f"del-{pid}", help="Remove from library"):
                core.remove_paper(collection, registry, pid)
                save_registry(registry, REGISTRY_PATH)
                st.rerun()

    st.divider()
    if st.session_state.messages and st.button("Clear conversation", use_container_width=True):
        clear_chat()
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main: the conversation
# --------------------------------------------------------------------------- #
st.title("The Research Synthesizer")
st.caption("Ask questions across your local paper library. Answers cite (Title, p.N).")

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        render_sources(msg.get('sources', []))

question = st.chat_input("Ask a question about your papers...")
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            # Everything before this turn is the conversational context.
            history = [
                {'role': m['role'], 'content': m['content']}
                for m in st.session_state.messages[:-1]
            ]
            try:
                result = core.answer_question(
                    collection, question, paper_id=scope_id, history=history
                )
            except Exception as e:
                st.error(f"Backend error: {e}\n\nIs Ollama running?")
                st.stop()

        if result['answer'] is None:
            answer = "I couldn't find anything relevant in your library for that question."
            st.markdown(answer)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': answer, 'sources': []}
            )
        else:
            st.markdown(result['answer'])
            render_sources(result['sources'])
            st.session_state.messages.append(
                {'role': 'assistant', 'content': result['answer'], 'sources': result['sources']}
            )

    save_chat(st.session_state.messages)
