"""
MMapReader — memory-mapped random-access reader for SHARD databases.

Core design principle:
  The shard file is NOT loaded into Python RAM.
  The OS maps the file into virtual address space via mmap() and pages in
  only the 4 KB blocks that contain the requested data.

This allows querying a 1 TB dataset on a machine with 2 GB of RAM:
  - Only the Bloom filter (~1 MB per shard) lives in RAM.
  - Only the OS-paged blocks of the accessed shard live in RAM.
  - Unaccessed shards contribute 0 bytes to RAM usage.

Lookup complexity:
  O(1) — FNV1a hash to shard index
  + O(k) — linear scan within one shard (k = records in that shard)

For well-tuned num_shards, k is small enough that linear scan is fast.
"""

import mmap
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from shard.core.sharding import ShardRouter
from shard.core.bloom_filter import BloomFilter
from shard.storage.binary_encoder import (
    decode_record,
    validate_header,
    HEADER_SIZE,
    _REC_LEN_FMT,
    _REC_LEN_SIZE,
)


class MMapReader:
    """
    Read-only, memory-mapped SHARD database reader.

    Usage::

        with MMapReader("./mydb", num_shards=1000) as reader:
            result = reader.find("ababol")
            if result:
                print(result)   # JSON string

    Lookup steps:
      1. Compute shard_id = FNV1a(key) % num_shards   [pure math, no I/O]
      2. Check BloomFilter[shard_id].contains(key)     [RAM only]
         → False: return None immediately (zero disk I/O)
         → True:  proceed
      3. mmap shard_xxxxxx.bin                         [OS-managed paging]
      4. Linear scan from HEADER_SIZE                  [page-level I/O]
      5. Return value on match, or None
    """

    def __init__(self, db_dir: str, num_shards: int = 1000, bloom_cache_size: int = 512) -> None:
        """
        Args:
            db_dir:          Path to the SHARD database directory.
            num_shards:      Number of shards the database was built with.
            bloom_cache_size: Maximum number of Bloom filters to keep in RAM
                             simultaneously. Oldest entries are evicted LRU-style
                             when the limit is reached.
                             At 1% FPR and 10k records/shard, each filter is ~12 KB.
                             512 filters ≈ 6 MB. Raise this on machines with more RAM.
        """
        self.db_dir = Path(db_dir)
        self.router = ShardRouter(num_shards)
        self._bloom_cache_size = bloom_cache_size
        self._mmaps: dict = {}
        self._file_handles: dict = {}
        self._blooms: OrderedDict = OrderedDict()  # LRU cache of shard_id → BloomFilter|None

    def find(self, key: str) -> Optional[str]:
        """
        Look up a key in the database.

        Args:
            key: The exact key to search for.

        Returns:
            The value string (typically JSON) if found, else None.
        """
        shard_id = self.router.get_shard(key)

        # Step 1: Bloom filter pre-check (RAM only, no disk I/O)
        bloom = self._get_bloom(shard_id)
        if bloom is not None and not bloom.contains(key):
            return None

        # Step 2: Memory-mapped scan
        mm = self._get_mmap(shard_id)
        if mm is None:
            return None

        return self._scan(mm, key)

    def close(self) -> None:
        """Release all memory maps and file handles."""
        for mm in self._mmaps.values():
            mm.close()
        for fh in self._file_handles.values():
            fh.close()
        self._mmaps.clear()
        self._file_handles.clear()

    def __enter__(self) -> "MMapReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_mmap(self, shard_id: int) -> Optional[mmap.mmap]:
        if shard_id not in self._mmaps:
            shard_path = self.db_dir / self.router.shard_filename(shard_id)
            if not shard_path.exists():
                return None
            fh = open(shard_path, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._file_handles[shard_id] = fh
            self._mmaps[shard_id] = mm
        return self._mmaps[shard_id]

    def _get_bloom(self, shard_id: int) -> Optional[BloomFilter]:
        if shard_id in self._blooms:
            # Move to end (most-recently-used)
            self._blooms.move_to_end(shard_id)
            return self._blooms[shard_id]

        bloom_path = self.db_dir / self.router.bloom_filename(shard_id)
        value = (
            BloomFilter.from_bytes(bloom_path.read_bytes())
            if bloom_path.exists()
            else None
        )

        self._blooms[shard_id] = value
        self._blooms.move_to_end(shard_id)

        # Evict least-recently-used entry if over capacity
        if len(self._blooms) > self._bloom_cache_size:
            self._blooms.popitem(last=False)

        return value

    def _scan(self, mm: mmap.mmap, target_key: str) -> Optional[str]:
        """
        Sequentially scan a shard for the target key.

        Starts after the 8-byte file header.  Uses the record_length prefix
        to jump from record to record without parsing every field.
        """
        offset = HEADER_SIZE
        mm_len = len(mm)

        while offset + _REC_LEN_SIZE <= mm_len:
            try:
                key, value, consumed = decode_record(mm, offset)
                if key == target_key:
                    return value
                offset += consumed
            except (struct.error, ValueError):
                # Truncated or corrupted record — stop scanning this shard
                break

        return None
