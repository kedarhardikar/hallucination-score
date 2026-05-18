"""
db.py
-----
Persistent vector store backed by ChromaDB (SQLite on disk).

ChromaDB stores everything in a local directory (default: ./chroma_db/).
Inside that directory you will find:
  - chroma.sqlite3          ← the main SQLite database
  - <uuid>/header.bin       ← binary HNSW index files (one per collection)
  - <uuid>/data_level0.bin

You can open chroma.sqlite3 with any SQLite viewer
(e.g. DB Browser for SQLite, DBeaver, or the VSCode SQLite extension)
and inspect tables:
  - collections             ← one row per collection
  - embeddings              ← one row per chunk (id, collection_id, embedding, document, metadata)
  - embedding_metadata      ← key/value metadata store

Public API
----------
  build_or_load_index(docs, persist_dir, collection_name, force_rebuild)
      → VectorStoreIndex
  inspect_db(persist_dir, collection_name)
      → prints a human-readable summary of every stored chunk
  reset_collection(persist_dir, collection_name)
      → deletes the collection so the next run rebuilds it
"""

import json
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Defaults ──────────────────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_docs"


# ── Build or load ─────────────────────────────────────────────────────────────
def build_or_load_index(
    docs=None,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    """
    If the collection already has chunks, load it directly (fast).
    If it is empty or force_rebuild=True, embed docs and persist (slow, one-time).

    Parameters
    ----------
    docs           : list of LlamaIndex Document objects (required on first run)
    persist_dir    : path to the ChromaDB directory
    collection_name: name of the Chroma collection
    force_rebuild  : if True, drop the existing collection and rebuild from docs
    """
    #1. Connect to ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)

    if force_rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"[DB] Dropped existing collection '{collection_name}'")
        except Exception:
            pass
    #3. Get or create collection in chroma
    collection   = client.get_or_create_collection(collection_name)
    #4. Wrap collection as vector store as llamaindex cannot directly connect to chroma
    vector_store = ChromaVectorStore(chroma_collection=collection)
    #5. Create storage context: tells llamaindex to "store all vectors in THIS vector DB"
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0 and not force_rebuild:
        print(f"[DB] Loaded '{collection_name}': {collection.count()} chunks "
              f"from {persist_dir}")
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        if not docs:
            raise ValueError(
                f"Collection '{collection_name}' is empty and no docs were provided. "
                "Pass docs= to build the index."
            )
        print(f"[DB] Building '{collection_name}' from {len(docs)} documents ...")
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
        print(f"[DB] Done — {collection.count()} chunks persisted to '{persist_dir}'")

    return index


# ── Inspect ───────────────────────────────────────────────────────────────────
def inspect_db(
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """
    Print a human-readable summary of every chunk stored in the collection.
    Embeddings are omitted (too long); use the SQLite file for raw vectors.
    """
    client      = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()

    print(f"\n{'='*60}")
    print(f"  ChromaDB path : {persist_dir}")
    print(f"  Collections   : {[c.name for c in collections]}")
    print(f"{'='*60}\n")

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        return

    total   = collection.count()
    results = collection.get(include=["documents", "metadatas"])

    print(f"Collection : '{collection_name}'")
    print(f"Chunks     : {total}\n")
    print(f"{'─'*60}")

    for i, (doc, meta) in enumerate(
        zip(results["documents"], results["metadatas"]), 1
    ):
        print(f"[{i:03d}] metadata : {meta}")
        preview = doc[:150].replace("\n", " ")
        print(f"       text     : {preview}{'...' if len(doc) > 150 else ''}")
        print()

    print(f"{'─'*60}")
    print(f"Total: {total} chunks\n")
    print(f"SQLite file: {persist_dir}/chroma.sqlite3")
    print("Open it with DB Browser for SQLite or any SQLite viewer.\n")

    return results


# ── Query cache (avoids re-downloading HotpotQA on every run) ────────────────
def save_queries(queries: list, collection_name: str, persist_dir: str = CHROMA_DIR):
    path = Path(persist_dir) / f"{collection_name}_queries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(queries, f)
    print(f"[DB] Queries cached: {path}")


def load_queries(collection_name: str, persist_dir: str = CHROMA_DIR):
    path = Path(persist_dir) / f"{collection_name}_queries.json"
    with open(path) as f:
        return json.load(f)


def queries_cached(collection_name: str, persist_dir: str = CHROMA_DIR) -> bool:
    return (Path(persist_dir) / f"{collection_name}_queries.json").exists()


# ── Existence check ───────────────────────────────────────────────────────────
def collection_exists(
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Returns True if the collection exists on disk and contains at least one chunk."""
    try:
        client     = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception:
        return False


# ── Reset ─────────────────────────────────────────────────────────────────────
def reset_collection(
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """Delete the collection so the next call to build_or_load_index rebuilds it."""
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
        print(f"[DB] Collection '{collection_name}' deleted from '{persist_dir}'")
    except Exception as e:
        print(f"[DB] Could not delete '{collection_name}': {e}")


# ── CLI — python db.py inspect / reset ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    col = sys.argv[2] if len(sys.argv) > 2 else COLLECTION_NAME

    if cmd == "inspect":
        inspect_db(collection_name=col)
    elif cmd == "reset":
        reset_collection(collection_name=col)
    else:
        print("Usage: python db.py [inspect|reset] [collection_name]")
