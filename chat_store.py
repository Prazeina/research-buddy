"""On-disk persistence for the web UI's conversation.

The whole reason chat memory survives a restart: Streamlit keeps the conversation
in RAM (st.session_state), which dies when the app closes. So we mirror it to a
file on disk and reload it on startup — exactly the same idea as papers.json keeps
your library from vanishing. Everything stays local to your machine.
"""
import json
import os

CHAT_PATH = 'chat_history.json'


def load_chat(path: str = CHAT_PATH) -> list:
    """Read the saved conversation. Returns [] if there's nothing saved yet
    (or the file is unreadable — a corrupt history shouldn't crash the app)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_chat(messages: list, path: str = CHAT_PATH) -> None:
    """Persist the full conversation to disk."""
    with open(path, 'w') as f:
        json.dump(messages, f, indent=2)


def clear_chat(path: str = CHAT_PATH) -> None:
    """Forget the saved conversation (the library is untouched)."""
    if os.path.exists(path):
        os.remove(path)
