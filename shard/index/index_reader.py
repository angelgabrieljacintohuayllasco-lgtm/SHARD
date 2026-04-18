"""
IndexReader — queries the MinHash similarity index.

Loads the pre-built index (index.minhash.bin + index.keymap.json) into RAM
and performs vectorized nearest-neighbor search using NumPy.

RAM usage after load:
  num_records × num_hashes × 4 bytes (signatures matrix)
  + num_records × ~30 bytes (key map, approximate)

For 1M records at 128 hashes:
  ~512 MB signatures + ~30 MB key map = ~542 MB total

For 100k records at 64 hashes:
  ~25 MB signatures + ~3 MB key map = ~28 MB total
"""

import json
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from shard.core.hasher import MinHasher


class IndexReader:
    """
    Loads and queries the MinHash similarity index.

    Given a query text, finds the most similar records by comparing
    MinHash signatures using vectorized NumPy operations.

    Usage::

        reader = IndexReader("./mydb")
        reader.load()
        results = reader.lookup("planta del campo", top_k=5)
        for key, score in results:
            print(f"{score:.3f}  {key}")
    """

    def __init__(self, db_dir: str) -> None:
        self.db_dir = Path(db_dir)
        self._signatures: Optional[np.ndarray] = None   # (N, H) uint32
        self._record_ids: Optional[np.ndarray] = None   # (N,) int64
        self._shard_ids: Optional[np.ndarray] = None    # (N,) int64
        self._keymap: dict = {}
        self._meta: dict = {}
        self._hasher: Optional[MinHasher] = None

    def load(self) -> "IndexReader":
        """
        Load the index from disk into RAM.

        Must be called before lookup().

        Returns:
            self (for chaining: reader.load().lookup(...))

        Raises:
            FileNotFoundError: If the index files are not found.
        """
        meta_path = self.db_dir / "index.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Index metadata not found in {self.db_dir}. "
                "Build the index first with IndexBuilder.build()."
            )

        with open(meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)

        num_hashes = self._meta["num_hashes"]
        self._hasher = MinHasher(num_hashes=num_hashes)

        # Load key map
        with open(self.db_dir / "index.keymap.json", encoding="utf-8") as f:
            self._keymap = {int(k): v for k, v in json.load(f).items()}

        # Load binary signatures
        index_path = self.db_dir / "index.minhash.bin"
        row_size = struct.calcsize(">II") + num_hashes * 4
        data = index_path.read_bytes()
        n = len(data) // row_size

        record_ids, shard_ids, signatures = [], [], []
        for i in range(n):
            chunk = data[i * row_size: (i + 1) * row_size]
            rid, sid = struct.unpack_from(">II", chunk, 0)
            sig = np.frombuffer(chunk[8:], dtype=">u4").astype(np.uint32)
            record_ids.append(rid)
            shard_ids.append(sid)
            signatures.append(sig)

        self._record_ids = np.array(record_ids, dtype=np.int64)
        self._shard_ids = np.array(shard_ids, dtype=np.int64)
        self._signatures = np.stack(signatures) if signatures else np.empty((0, num_hashes), dtype=np.uint32)

        return self

    def lookup(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top_k records most similar to the query text.

        Similarity is estimated as the fraction of MinHash positions where
        the query signature matches the stored signature (Jaccard estimate).

        Args:
            query_text: The text to search for (natural language).
            top_k:      Maximum number of results to return.

        Returns:
            List of (key, similarity_score) tuples, highest score first.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if self._signatures is None:
            raise RuntimeError("Index not loaded. Call .load() first.")

        if len(self._signatures) == 0:
            return []

        query_sig = self._hasher.signature(query_text)
        # Vectorized Jaccard estimate: fraction of equal positions per row
        similarities = np.mean(self._signatures == query_sig[np.newaxis, :], axis=1)

        actual_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:actual_k]

        results = []
        for idx in top_indices:
            rid = int(self._record_ids[idx])
            score = float(similarities[idx])
            key = self._keymap.get(rid, f"record_{rid}")
            results.append((key, score))

        return results

    @property
    def record_count(self) -> int:
        """Number of records loaded in the index."""
        if self._record_ids is None:
            return 0
        return len(self._record_ids)

    @property
    def metadata(self) -> dict:
        """Index build metadata (num_shards, num_hashes, total_records)."""
        return self._meta.copy()
