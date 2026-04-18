"""
Bloom Filter — probabilistic membership test for SHARD.

Used as a pre-filter before opening shard files:
  - If BloomFilter says NO → the key is definitely absent. Skip disk I/O.
  - If BloomFilter says YES → the key is probably present. Open the shard.

False negative rate: 0% (guaranteed by design).
False positive rate: configurable. Default 1% (1 false alarm per 100 probes).

Memory formula for capacity C and false positive rate p:
  bits = -(C × ln(p)) / (ln(2)²)
  hashes = round(bits/C × ln(2))

At 1% FPR, 1M items requires ~1.2 MB of RAM.
The filter is serialized to a .bloom companion file alongside each shard.
"""

import math
import hashlib
import struct


class BloomFilter:
    """
    Space-efficient probabilistic set membership structure.

    Guarantees:
    - Zero false negatives: if a key was added, contains() always returns True.
    - Low false positives: non-added keys return True with probability ≤ fpr.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        false_positive_rate: float = 0.01,
    ) -> None:
        self.capacity = capacity
        self.fpr = false_positive_rate
        self.num_bits = self._optimal_bits(capacity, false_positive_rate)
        self.num_hashes = self._optimal_hashes(self.num_bits, capacity)
        self._bits = bytearray(math.ceil(self.num_bits / 8))
        self._count = 0

    def add(self, key: str) -> None:
        """Add a key to the filter."""
        for idx in self._hash_positions(key):
            self._bits[idx >> 3] |= (1 << (idx & 7))
        self._count += 1

    def contains(self, key: str) -> bool:
        """
        Test membership.

        Returns:
            False → key was definitely NOT added (zero false negatives).
            True  → key was probably added (may be a false positive).
        """
        return all(
            (self._bits[idx >> 3] >> (idx & 7)) & 1
            for idx in self._hash_positions(key)
        )

    def to_bytes(self) -> bytes:
        """Serialize the filter for storage as a companion .bloom file."""
        header = struct.pack(">IId", self.capacity, self.num_bits, self.fpr)
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserialize a filter from bytes read from a .bloom file."""
        header_size = struct.calcsize(">IId")
        capacity, num_bits, fpr = struct.unpack(">IId", data[:header_size])
        bf = cls(capacity=capacity, false_positive_rate=fpr)
        bf._bits = bytearray(data[header_size:])
        return bf

    @property
    def count(self) -> int:
        """Approximate number of items added."""
        return self._count

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set to 1. Approaches 0.5 at optimal fill."""
        set_bits = sum(bin(b).count("1") for b in self._bits)
        return set_bits / self.num_bits

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _hash_positions(self, key: str):
        """
        Generate num_hashes independent bit positions using double hashing.

        Double hashing: position_i = (h1 + i * h2) % num_bits
        Uses MD5 and SHA-1 as the two base hashes.
        """
        data = key.encode("utf-8")
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        h2 = int(hashlib.sha1(data).hexdigest(), 16)
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    @staticmethod
    def _optimal_bits(n: int, p: float) -> int:
        """Optimal number of bits: m = -(n·ln(p)) / (ln(2)²)"""
        return math.ceil(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Optimal number of hash functions: k = round(m/n · ln(2))"""
        return max(1, round(m / n * math.log(2)))
