"""Optional semantic recall backfill — v0.3.1 example (GGUF model).

memory-arbiter does NOT bundle an embedding model. This script shows how to
generate embeddings with a local GGUF model (loaded via llama-cpp-python) and
backfill them into the shared SQLite so that `memory_search(query_embedding=...)`
can do semantic recall.

This is model-agnostic in principle — swap the model_path / embed call for any
other backend (sentence-transformers, OpenAI API, Ollama, etc.) and the rest
of the flow is identical. See README "Optional: Semantic Recall" for context.

Note: this example ships the GGUF path because it's the lightest dependency
(llama-cpp-python, ~17MB) and reuses GGUF models you may already have. For
sentence-transformers, replace `load_gguf_embedder` with a SentenceTransformer
encode call — the rest of backfill/search is identical. The README documents
all three paths.

Install:
    pip install llama-cpp-python        # lightweight, no model bundled
    # Then point MODEL_PATH at any GGUF embedding model you already have.

Usage:
    # Set the model path (or rely on the default below).
    export MEMORY_ARBITER_GGUF=/path/to/embedding-model.gguf
    # Make sure MEMORY_ARBITER_VEC_DIM matches the model's output dim.
    python docs/semantic_example.py            # backfill all active memories
    python docs/semantic_example.py --query "金营平台营销"   # try a semantic search
    python docs/semantic_example.py --query "金营平台营销" --compare  # with/without vec

Why a separate script instead of auto-embedding on every write?
  - Keeps the core package dependency-free and lightweight.
  - You choose the model, the runtime (GGUF/ONNX/API), the cost.
  - Re-running this script refreshes embeddings after schema/model changes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402

# Default model path: reuses the OpenClaw-installed embeddinggemma GGUF if
# present, so you don't need to download anything new. Override with
# MEMORY_ARBITER_GGUF env var or --model flag.
DEFAULT_GGUF = os.path.expanduser(
    "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"
)


def load_gguf_embedder(model_path: str):
    """Load a GGUF embedding model via llama-cpp-python.

    Returns encode_fn(text) -> list[float]. Raises if llama-cpp-python
    is missing or the model file cannot be loaded.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "llama-cpp-python not installed.\n  pip install llama-cpp-python",
            file=sys.stderr,
        )
        raise
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    llm = Llama(model_path=model_path, embedding=True, verbose=False)

    def encode(text: str) -> list[float]:
        data = llm.create_embedding(text)["data"][0]["embedding"]
        return [float(x) for x in data]

    # Return the model's actual output dim for sanity-checking against config.
    sample = encode("dimension probe")
    return encode, len(sample)


def backfill(tools: MemoryTools, encode_fn) -> tuple[int, float]:
    """Embed every active memory's (subject + content) and store it.

    Returns (count, elapsed_seconds). Re-runs replace existing embeddings
    (store_embedding deletes-then-inserts), so this is idempotent.
    """
    db: MemoryDB = tools.db
    if db.conn is None:
        raise RuntimeError("DB connection unavailable")
    rows = db.conn.execute(
        "SELECT id, subject, content FROM memories WHERE status = 'active' ORDER BY id"
    ).fetchall()
    t0 = time.time()
    count = 0
    for row in rows:
        rid = row["id"]
        text = f"{row['subject'] or ''}\n{row['content'] or ''}".strip() or "(empty)"
        vec = encode_fn(text)
        result = tools.memory_store_embedding(memory_id=rid, embedding=vec)
        if result.get("ok") and result.get("data", {}).get("stored"):
            count += 1
    return count, time.time() - t0


def semantic_search(tools: MemoryTools, encode_fn, query: str, limit: int = 5):
    qvec = encode_fn(query)
    return tools.memory_search(query=query, query_embedding=qvec, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="memory-arbiter semantic backfill / search (GGUF)")
    parser.add_argument("--query", default=None, help="if set, run a semantic search instead of backfill")
    parser.add_argument("--compare", action="store_true", help="with --query: print lexical-only vs semantic results side by side")
    parser.add_argument("--model", default=os.getenv("MEMORY_ARBITER_GGUF", DEFAULT_GGUF), help="path to GGUF embedding model")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading GGUF model: {args.model}", file=sys.stderr)
    encode_fn, model_dim = load_gguf_embedder(args.model)
    print(f"Model output dim: {model_dim}", file=sys.stderr)

    settings = Settings.from_env()
    if settings.vec_dim != model_dim:
        print(
            f"WARNING: MEMORY_ARBITER_VEC_DIM={settings.vec_dim} but model outputs "
            f"dim={model_dim}. Set the env var to match and recreate memories_vec, "
            "or KNN will error at query time.",
            file=sys.stderr,
        )

    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    if args.query:
        if args.compare:
            # Lexical only (v0.3.0 behaviour).
            print("\n=== Lexical only (no embedding) ===")
            lex = tools.memory_search(query=args.query, limit=args.limit)
            for r in lex.get("data", {}).get("results", []):
                print(f"  [{r.get('id')}] {r.get('subject') or '(no subject)'}")
            # Semantic (v0.3.1).
            print("\n=== With semantic recall ===")
        result = semantic_search(tools, encode_fn, args.query, limit=args.limit)
        print(f"\nSemantic search for: {args.query!r}")
        for r in result.get("data", {}).get("results", []):
            print(f"  [{r.get('id')}] {r.get('subject') or '(no subject)'}")
            print(f"      { (r.get('content') or '')[:80] }...")
    else:
        count, elapsed = backfill(tools, encode_fn)
        print(f"Backfilled {count} memories with embeddings (dim={model_dim}) in {elapsed:.1f}s.")
        print("Now call memory_search with query_embedding=... to enable semantic recall.")


if __name__ == "__main__":
    main()
