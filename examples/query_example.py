"""
SHARD Query Example
===================
Demonstrates exact key lookup and similarity search on a SHARD database.

Run build_from_json.py first to generate the demo database.

Run:
    python examples/build_from_json.py
    python examples/query_example.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shard.storage.mmap_reader import MMapReader
from shard.index.index_reader import IndexReader

DB_DIR = os.path.join(os.path.dirname(__file__), "demo_db")
NUM_SHARDS = 16


def section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def exact_lookup(reader: MMapReader, key: str) -> None:
    result = reader.find(key)
    if result:
        record = json.loads(result)
        print(f"  ✓  {record['lemma']}")
        print(f"     {record['definition']}")
    else:
        print(f"  ✗  Key not found: {key!r}")


def similarity_search(index: IndexReader, query: str, top_k: int = 4) -> None:
    results = index.lookup(query, top_k=top_k)
    if results:
        for key, score in results:
            print(f"  {score:.4f}  {key}")
    else:
        print("  (no results)")


def main() -> None:
    if not os.path.isdir(DB_DIR):
        print(f"Database not found at: {DB_DIR}")
        print("Run 'python examples/build_from_json.py' first.")
        sys.exit(1)

    # ── Exact lookup ───────────────────────────────────────────────────────────
    section("Exact Lookup (mmap, zero full-file load)")

    with MMapReader(DB_DIR, num_shards=NUM_SHARDS) as reader:
        print(f"\nQuery: 'ababol'")
        exact_lookup(reader, "ababol")

        print(f"\nQuery: 'ábaco'")
        exact_lookup(reader, "ábaco")

        print(f"\nQuery: 'abadía'")
        exact_lookup(reader, "abadía")

        print(f"\nQuery: 'palabra_inexistente'")
        exact_lookup(reader, "palabra_inexistente")

    # ── Semantic similarity search ─────────────────────────────────────────────
    section("Semantic Similarity Search (MinHash index)")

    index = IndexReader(DB_DIR)
    index.load()
    print(f"\nIndex loaded: {index.record_count} records")

    print(f"\nQuery: 'planta silvestre del campo'")
    similarity_search(index, "planta silvestre del campo")

    print(f"\nQuery: 'pez marino comestible'")
    similarity_search(index, "pez marino comestible")

    print(f"\nQuery: 'instrumento de cálculo matemático'")
    similarity_search(index, "instrumento de cálculo matemático")

    print(f"\nQuery: 'monasterio religiosa convento'")
    similarity_search(index, "monasterio religiosa convento")

    section("Done")
    print("\nThese queries ran on disk-backed binary files.")
    print("No JSON parsing. No SQL. Just hash + mmap.")


if __name__ == "__main__":
    main()
