"""
IndexBuilder — constructs the MinHash similarity index for a SHARD database.

The index enables approximate semantic search: find records similar to a
query even when the exact key is unknown.

Index files written to the database directory:
  index.minhash.bin   — flat binary array of (record_id, shard_id, sig[H])
  index.keymap.json   — {record_id: key} mapping for result retrieval
  index.meta.json     — build parameters: num_shards, num_hashes, total_records

RAM during build: O(total_records × num_hashes × 4 bytes)
  For 1M records at 128 hashes: ~512 MB
  For 100k records at 64 hashes: ~25 MB

Records are accumulated in memory, then flushed to disk with .build().
For very large datasets (> RAM capacity), build the index in batches
by calling .build() after each batch and merging the binary files.
"""

import json
import struct
from pathlib import Path
from typing import Iterable, Tuple, Dict

import numpy as np

from shard.core.hasher import MinHasher
from shard.core.sharding import ShardRouter


class IndexBuilder:
    """
    Builds the MinHash similarity index from (record_id, key, text) triples.

    Usage::

        builder = IndexBuilder("./mydb", num_shards=1000, num_hashes=128)
        for i, (key, text) in enumerate(my_records):
            builder.add(i, key, text)
        builder.build()
    """

    def __init__(
        self,
        db_dir: str,
        num_shards: int = 1000,
        num_hashes: int = 128,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.router = ShardRouter(num_shards)
        self.hasher = MinHasher(num_hashes=num_hashes)
        self.num_hashes = num_hashes
        self._entries: list = []         # List of (record_id, shard_id, sig)
        self._keymap: Dict[int, str] = {}

    def add(self, record_id: int, key: str, text: str) -> None:
        """
        Add a single record to the index.

        Args:
            record_id: Unique integer identifier for the record.
            key:       The record's lookup key (used for retrieval).
            text:      The full text to hash (key + definition, or similar).
        """
        sig = self.hasher.signature(text)
        shard_id = self.router.get_shard(key)
        self._entries.append((record_id, shard_id, sig))
        self._keymap[record_id] = key

    def add_batch(self, records: Iterable[Tuple[int, str, str]]) -> None:
        """
        Add multiple records.

        Args:
            records: Iterable of (record_id, key, text) tuples.
        """
        for record_id, key, text in records:
            self.add(record_id, key, text)

    def build(self) -> None:
        """
        Write the index to disk.

        Creates three files in db_dir:
          - index.minhash.bin: flat binary array, row = [4B record_id][4B shard_id][H×4B sig]
          - index.keymap.json: {record_id: key} mapping
          - index.meta.json:   build metadata
        """
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Write MinHash binary index
        index_path = self.db_dir / "index.minhash.bin"
        with open(index_path, "wb") as f:
            for record_id, shard_id, sig in self._entries:
                row = (
                    struct.pack(">II", record_id, shard_id)
                    + sig.astype(">u4").tobytes()
                )
                f.write(row)

        # Write key map
        keymap_path = self.db_dir / "index.keymap.json"
        with open(keymap_path, "w", encoding="utf-8") as f:
            json.dump(self._keymap, f, ensure_ascii=False)

        # Write metadata
        meta_path = self.db_dir / "index.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_shards": self.router.num_shards,
                    "num_hashes": self.num_hashes,
                    "total_records": len(self._entries),
                },
                f,
            )

    @property
    def record_count(self) -> int:
        """Number of records currently accumulated in the builder."""
        return len(self._entries)
