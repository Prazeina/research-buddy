import argparse
import json
import os
import sys

import core
from rag_utils import (
    load_registry,
    missing_ollama_models,
    save_registry,
)

REQUIRED_MODELS = ['llama3.2', 'nomic-embed-text']
REGISTRY_PATH = 'papers.json'


def check_prereqs() -> None:
    """Validate Ollama models are installed. Exits with a friendly message if not."""
    missing = missing_ollama_models(REQUIRED_MODELS)
    if missing:
        print("\nRequired Ollama models are not installed (or Ollama isn't running):")
        for name in missing:
            print(f"   - {name}    ->  ollama pull {name}")
        sys.exit(1)


def connect_chroma_or_exit():
    """Return a Chroma client, or exit if the server isn't reachable."""
    try:
        return core.connect_chroma()
    except Exception:
        print("\nCould not connect to ChromaDB server at localhost:8000.")
        print("Start it in another terminal with:  chroma run --path ./chroma_db")
        sys.exit(1)


def load_registry_or_exit(path: str = REGISTRY_PATH) -> dict:
    """Load the registry; exit with a friendly message if it's malformed."""
    try:
        return load_registry(path)
    except json.JSONDecodeError as e:
        print(f"papers.json is corrupt: {e}")
        print(f"Fix it by hand, or delete it to start a fresh registry.")
        sys.exit(1)


def resolve_paper_id(registry: dict, prefix: str) -> str:
    """Match a paper_id prefix to a full ID. Exits on no match or ambiguous match."""
    matches = [pid for pid in registry if pid.startswith(prefix)]
    if not matches:
        print(f"No paper matches prefix '{prefix}'. Try `rag list` to see all paper IDs.")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Prefix '{prefix}' is ambiguous. Matches:")
        for pid in sorted(matches):
            print(f"   {pid}  {registry[pid]['title']}")
        print(f"Use a longer prefix.")
        sys.exit(1)
    return matches[0]


def backend_op(description: str, fn):
    """Run a Chroma/Ollama-backed call with a clean error if a service dies.

    Many operations (queries, adds) depend on BOTH ChromaDB and Ollama being up,
    so the error message points the user to check both rather than guessing.
    """
    try:
        return fn()
    except Exception as e:
        print(f"\nBackend error during {description}: {e}")
        print(f"Check that ChromaDB (localhost:8000) and Ollama are both still running.")
        sys.exit(1)


def open_collection(client):
    """Open the shared library collection, or exit cleanly on failure."""
    return backend_op(
        "opening the library collection",
        lambda: core.get_library_collection(client),
    )


def _print_ingest_result(result: dict, verbose: bool) -> None:
    """Render an ingest result dict from core.ingest_pdf as CLI output."""
    status = result['status']
    basename = result['filename']

    if status == 'skipped':
        if verbose:
            print(f"Already in library -- skipping.")
            print(f"   paper_id: {result['paper_id']}")
            print(f"   title:    {result['title']}")
        else:
            print(f"  [skip] {basename}  ({result['title']})")
    elif status == 'failed':
        if verbose:
            print(result['error'])
        else:
            print(f"  [fail] {basename}  ({result['error']})")
    else:  # added
        if verbose:
            print(f"\nAdded to library.")
            print(f"   paper_id: {result['paper_id']}")
            print(f"   title:    {result['title']}")
            print(f"   chunks:   {result['n_chunks']} across {result['n_pages']} pages")
        else:
            print(f"  [add ] {basename}  ({result['n_chunks']} chunks)  -> {result['title']}")


def cmd_add(args: argparse.Namespace) -> None:
    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        print(f"Not found: {pdf_path}")
        sys.exit(1)

    check_prereqs()
    registry = load_registry_or_exit()
    client = connect_chroma_or_exit()
    collection = open_collection(client)

    if os.path.isdir(pdf_path):
        if args.recursive:
            pdf_paths = [
                os.path.join(root, f)
                for root, _, files in os.walk(pdf_path)
                for f in files
                if f.lower().endswith('.pdf')
            ]
        else:
            pdf_paths = [
                os.path.join(pdf_path, f)
                for f in os.listdir(pdf_path)
                if f.lower().endswith('.pdf')
            ]
        pdf_paths.sort()

        if not pdf_paths:
            scope = "or its subfolders" if args.recursive else ""
            print(f"No PDFs found in {pdf_path}{scope}.")
            sys.exit(1)

        if args.title:
            print(f"Note: --title is ignored in batch mode (titles auto-detect per file)\n")

        print(f"Found {len(pdf_paths)} PDF(s)\n")
        counts = {'added': 0, 'skipped': 0, 'failed': 0}
        for path in pdf_paths:
            result = backend_op(
                f"ingesting {os.path.basename(path)}",
                lambda p=path: core.ingest_pdf(p, collection, registry),
            )
            counts[result['status']] += 1
            _print_ingest_result(result, verbose=False)
            if result['status'] == 'added':
                save_registry(registry, REGISTRY_PATH)
        print(
            f"\nBatch complete: {counts['added']} added, "
            f"{counts['skipped']} skipped, {counts['failed']} failed."
        )
    else:
        print(f"Reading {pdf_path}...")
        result = backend_op(
            f"ingesting {os.path.basename(pdf_path)}",
            lambda: core.ingest_pdf(pdf_path, collection, registry, title_override=args.title),
        )
        _print_ingest_result(result, verbose=True)
        if result['status'] == 'added':
            save_registry(registry, REGISTRY_PATH)
        elif result['status'] == 'failed':
            sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    registry = load_registry(REGISTRY_PATH)
    if not registry:
        print("No papers in the library yet.")
        print("Add one with:  rag add <path-to-pdf>")
        return

    items = sorted(
        registry.items(),
        key=lambda kv: kv[1]['added_at'],
        reverse=True,
    )

    print(f"{'PAPER_ID':<10} {'TITLE':<60} ADDED")
    print(f"{'-' * 8:<10} {'-' * 58:<60} {'-' * 10}")
    for paper_id, meta in items:
        title = meta['title']
        if len(title) > 58:
            title = title[:57] + '...'
        added = meta['added_at'][:10]
        print(f"{paper_id:<10} {title:<60} {added}")

    print(f"\n{len(items)} paper{'s' if len(items) != 1 else ''} in library.")


def cmd_ask(args: argparse.Namespace) -> None:
    check_prereqs()

    paper_id_full = None
    if args.paper:
        registry = load_registry_or_exit()
        paper_id_full = resolve_paper_id(registry, args.paper)
        if paper_id_full != args.paper:
            print(f"Scoped to: {registry[paper_id_full]['title']} ({paper_id_full})\n")

    client = connect_chroma_or_exit()
    collection = open_collection(client)

    if backend_op("counting library", collection.count) == 0:
        print("Library is empty. Add a paper first with:  rag add <pdf>")
        sys.exit(1)

    print("Thinking...\n")
    result = backend_op(
        "querying the library",
        lambda: core.answer_question(
            collection, args.question, paper_id=paper_id_full, top=args.top
        ),
    )

    if result['answer'] is None:
        scope = f"paper_id={args.paper}" if args.paper else "the library"
        print(f"No matching chunks found in {scope}.")
        return

    print(result['answer'])


def cmd_remove(args: argparse.Namespace) -> None:
    registry = load_registry_or_exit()
    paper_id = resolve_paper_id(registry, args.paper)
    title = registry[paper_id]['title']

    client = connect_chroma_or_exit()
    collection = open_collection(client)

    chunk_count = backend_op(
        f"removing {paper_id}",
        lambda: core.remove_paper(collection, registry, paper_id),
    )
    save_registry(registry, REGISTRY_PATH)

    print(f"Removed: {title} ({paper_id})")
    print(f"   {chunk_count} chunks deleted from Chroma")
    print(f"   registry entry removed")


def cmd_retitle(args: argparse.Namespace) -> None:
    registry = load_registry_or_exit()
    paper_id = resolve_paper_id(registry, args.paper)
    old_title = registry[paper_id]['title']
    new_title = args.title

    if old_title == new_title:
        print(f"Title unchanged.")
        return

    client = connect_chroma_or_exit()
    collection = open_collection(client)

    chunks = backend_op(
        f"loading chunks for {paper_id}",
        lambda: collection.get(where={'paper_id': paper_id}, include=['metadatas']),
    )

    if not chunks['ids']:
        print(f"Inconsistent state: registry has {paper_id} but Chroma has no chunks for it.")
        print(f"Run `rag remove {paper_id}` to drop the orphan registry entry, then re-add the PDF.")
        sys.exit(1)

    new_metadatas = [{**meta, 'title': new_title} for meta in chunks['metadatas']]

    backend_op(
        "updating chunk titles",
        lambda: collection.update(ids=chunks['ids'], metadatas=new_metadatas),
    )

    registry[paper_id]['title'] = new_title
    save_registry(registry, REGISTRY_PATH)

    print(f"Retitled {paper_id}:")
    print(f"   from: {old_title}")
    print(f"   to:   {new_title}")
    print(f"   ({len(chunks['ids'])} chunk metadatas updated)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='rag',
        description='The Research Synthesizer -- local RAG over your paper library',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    add_p = sub.add_parser('add', help='ingest a PDF (or a directory of PDFs) into the library')
    add_p.add_argument('pdf', help='path to a PDF file, or a directory containing PDFs')
    add_p.add_argument('--title', help='override the auto-detected title (single-file only)')
    add_p.add_argument('-r', '--recursive', action='store_true', help='when given a directory, recurse into subfolders')
    add_p.set_defaults(func=cmd_add)

    list_p = sub.add_parser('list', help='show papers in the library')
    list_p.set_defaults(func=cmd_list)

    ask_p = sub.add_parser('ask', help='ask a question across the library (or a single paper)')
    ask_p.add_argument('question', help='the question to ask')
    ask_p.add_argument('--paper', help='scope the question to one paper_id (prefix ok)')
    ask_p.add_argument('--top', type=int, default=5, help='number of chunks to retrieve (default 5)')
    ask_p.set_defaults(func=cmd_ask)

    remove_p = sub.add_parser('remove', help='remove a paper from the library')
    remove_p.add_argument('paper', help='paper_id (or unique prefix) to remove')
    remove_p.set_defaults(func=cmd_remove)

    retitle_p = sub.add_parser('retitle', help='change the title stored for a paper')
    retitle_p.add_argument('paper', help='paper_id (or unique prefix)')
    retitle_p.add_argument('title', help='the new title')
    retitle_p.set_defaults(func=cmd_retitle)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
