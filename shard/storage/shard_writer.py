"""
ShardWriter — streaming builder for SHARD binary databases.

Converts an iterable of (key, value) pairs into sharded binary files
without loading the full dataset into memory.

Memory footprint during build:
  - O(open_shards) file handles (at most num_shards, but typically few)
  - O(bloom_capacity) per open Bloom filter
  - Input records are not accumulated in RAM — processed one at a time.

This makes ShardWriter suitable for streaming TB-scale datasets through
a machine with limited RAM.
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

from shard.core.sharding import ShardRouter
from shard.core.bloom_filter import BloomFilter
from shard.storage.binary_encoder import encode_record, shard_file_header


class ShardWriter:
    """
    Context-manager-based writer for SHARD databases.

    Usage::

        with ShardWriter("./mydb", num_shards=1000) as writer:
            for key, value in my_data_stream:
                writer.write(key, value)

    On exit, all open shard files are closed and their Bloom filters
    are written to companion .bloom files.

    ``estimated_total_records`` is used to size the per-shard Bloom filter so
    its false-positive rate stays at ``bloom_fpr`` even when shards fill up.
    When not provided, ``bloom_capacity`` is used directly.
    """

    def __init__(
        self,
        output_dir: str,
        num_shards: int = 1000,
        bloom_capacity: int = 10_000,
        bloom_fpr: float = 0.01,
        estimated_total_records: Optional[int] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.router = ShardRouter(num_shards)
        self.bloom_fpr = bloom_fpr
        self._handles: Dict[int, object] = {}
        self._blooms: Dict[int, BloomFilter] = {}
        self._total_written = 0

        # Auto-calculate per-shard capacity when total is known.
        # Add 20% headroom to handle non-uniform key distribution.
        if estimated_total_records is not None and estimated_total_records > 0:
            per_shard = max(1, estimated_total_records // num_shards)
            self.bloom_capacity = int(per_shard * 1.2)
        else:
            self.bloom_capacity = bloom_capacity

    def __enter__(self) -> "ShardWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def write(self, key: str, value: str) -> None:
        """
        Write a single key-value record to the appropriate shard.

        The shard is determined by: shard_id = FNV1a(key) % num_shards.
        The Bloom filter for that shard is updated with the key.

        Args:
            key:   Record key (any UTF-8 string).
            value: Record value (any UTF-8 string, typically JSON).
        """
        shard_id = self.router.get_shard(key)
        fh = self._get_file_handle(shard_id)
        fh.write(encode_record(key, value))
        self._blooms[shard_id].add(key)
        self._total_written += 1

    def write_batch(self, records: Iterable[Tuple[str, str]]) -> int:
        """
        Stream-write multiple records.

        Args:
            records: Iterable of (key, value) tuples.

        Returns:
            Number of records written in this batch.
        """
        count = 0
        for key, value in records:
            self.write(key, value)
            count += 1
        return count

    def close(self) -> None:
        """Flush and close all open shard files; write Bloom filters to disk."""
        for shard_id, fh in self._handles.items():
            fh.close()
            bloom_path = self.output_dir / self.router.bloom_filename(shard_id)
            bloom_path.write_bytes(self._blooms[shard_id].to_bytes())
        self._handles.clear()
        self._blooms.clear()

    @property
    def total_written(self) -> int:
        """Total number of records written since this writer was created."""
        return self._total_written

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_file_handle(self, shard_id: int):
        if shard_id not in self._handles:
            path = self.output_dir / self.router.shard_filename(shard_id)
            fh = open(path, "ab")
            # Write header only for new (empty) files
            if fh.tell() == 0:
                fh.write(shard_file_header())
            self._handles[shard_id] = fh
            self._blooms[shard_id] = BloomFilter(self.bloom_capacity, self.bloom_fpr)
        return self._handles[shard_id]
