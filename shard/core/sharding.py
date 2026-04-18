"""
ShardRouter — maps record keys to shard file numbers.

Uses FNV-1a (Fowler–Noll–Vo) hashing, chosen over Python's built-in hash()
because:
  - Deterministic across all Python versions and platforms.
  - Not subject to Python's hash randomization (PYTHONHASHSEED).
  - Extremely fast: no cryptographic overhead.
  - Excellent distribution for short strings (dictionary keys, words, IDs).

The routing formula is:  shard_id = FNV1a(key) % num_shards

This means a key ALWAYS maps to the same shard file, making lookup O(1)
before even touching disk.
"""


class ShardRouter:
    """
    Deterministic key-to-shard mapper.

    Given a key (string), returns the shard index [0, num_shards) where
    that key's record is (or should be) stored.

    The same key always maps to the same shard, on any machine, in any
    Python version, with any PYTHONHASHSEED setting.
    """

    # FNV-1a 64-bit constants
    _FNV_PRIME = 0x00000100000001B3
    _FNV_OFFSET = 0xCBF29CE484222325

    def __init__(self, num_shards: int = 1000) -> None:
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        self.num_shards = num_shards

    def get_shard(self, key: str) -> int:
        """
        Return the shard index for a given key.

        Args:
            key: The record key (e.g. a word, ID, or term).

        Returns:
            Integer in [0, num_shards).
        """
        return self._fnv1a(key.encode("utf-8")) % self.num_shards

    def shard_filename(self, shard_id: int) -> str:
        """Return the standardized filename for a shard index."""
        return f"shard_{shard_id:06d}.bin"

    def bloom_filename(self, shard_id: int) -> str:
        """Return the Bloom filter companion filename for a shard."""
        return f"shard_{shard_id:06d}.bloom"

    def _fnv1a(self, data: bytes) -> int:
        """FNV-1a 64-bit hash. Returns unsigned 64-bit integer."""
        h = self._FNV_OFFSET
        for byte in data:
            h ^= byte
            h = (h * self._FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        return h

    @classmethod
    def recommended_shards(
        cls,
        estimated_records: int,
        avg_record_bytes: int = 512,
        target_shard_bytes: int = 4 * 1024 * 1024,
    ) -> int:
        """
        Heuristic: choose num_shards so each shard is ~target_shard_bytes.

        Default target is 4 MB per shard — a good balance between:
        - Low overhead (avoid millions of tiny files).
        - Fast linear scan within a shard (keep shards small).

        Returns the next power of 2 >= the computed recommendation,
        which ensures even modulo distribution.
        """
        total_bytes = estimated_records * avg_record_bytes
        recommended = max(1, total_bytes // target_shard_bytes)
        # Round up to next power of 2
        p = 1
        while p < recommended:
            p *= 2
        return p
