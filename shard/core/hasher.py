"""
MinHash and SimHash implementations for SHARD.

These are the two locality-sensitive hashing algorithms that power
SHARD's similarity search index:

- MinHash: Estimates Jaccard similarity between documents.
  A database of 1M records can be indexed in ~512 MB (128 hashes × 4 bytes × 1M).

- SimHash: Produces a 64-bit fingerprint where similar texts have
  low Hamming distance. Near-duplicate detection in O(1) space per record.
"""

import hashlib
import math
from typing import Set

import numpy as np


class MinHasher:
    """
    MinHash signature generator for approximate nearest-neighbor search.

    Converts a text document into a compact signature array (List[uint32]).
    Two documents with similar MinHash signatures have high Jaccard similarity.

    Memory usage per signature: num_hashes × 4 bytes.
    Default (128 hashes) = 512 bytes per record.

    Theory: For k-shingle sets A and B:
        P(MinHash_i(A) == MinHash_i(B)) = Jaccard(A, B) = |A∩B| / |A∪B|

    The signature is the array [min_hash_1, ..., min_hash_k] — estimating
    Jaccard by counting equal positions in two signatures.
    """

    def __init__(self, num_hashes: int = 128, seed: int = 42) -> None:
        self.num_hashes = num_hashes
        rng = np.random.RandomState(seed)
        # Pre-generate universal hash function parameters (a, b) with
        # Carter-Wegman family: h(x) = (a*x + b) mod p
        self._a = rng.randint(1, 2**31 - 1, size=num_hashes, dtype=np.int64)
        self._b = rng.randint(0, 2**31 - 1, size=num_hashes, dtype=np.int64)
        self._p = (2**61) - 1  # Mersenne prime — fast modulo

    def signature(self, text: str) -> np.ndarray:
        """
        Compute the MinHash signature of a text string.

        Args:
            text: Input document (any language, any encoding).

        Returns:
            Signature array of shape (num_hashes,), dtype uint32.
        """
        shingles = self._shingle(text, k=3)
        if not shingles:
            return np.zeros(self.num_hashes, dtype=np.uint32)

        sig = np.full(self.num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)

        for shingle in shingles:
            h = int(hashlib.sha1(shingle.encode("utf-8", errors="replace")).hexdigest(), 16) % self._p
            vals = (self._a * h + self._b) % self._p
            sig = np.minimum(sig, vals)

        return sig.astype(np.uint32)

    def similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from two MinHash signatures.

        Returns a value in [0, 1]. Two identical texts → 1.0.
        Two completely different texts → ~0.0.
        """
        return float(np.mean(sig_a == sig_b))

    def to_bytes(self, sig: np.ndarray) -> bytes:
        """Serialize a signature to big-endian bytes."""
        return sig.astype(">u4").tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        """Deserialize a signature from big-endian bytes."""
        return np.frombuffer(data, dtype=">u4").astype(np.uint32)

    @staticmethod
    def _shingle(text: str, k: int = 3) -> Set[str]:
        """
        Build the k-shingle set (k consecutive words) from text.
        Shingles capture local word context for similarity estimation.
        """
        text = text.lower().strip()
        words = text.split()
        if len(words) < k:
            return set(words)
        return {" ".join(words[i: i + k]) for i in range(len(words) - k + 1)}


class SimHasher:
    """
    SimHash (Charikar, 2002) — locality-sensitive fingerprinting.

    Maps a text to a fixed-size binary integer (fingerprint) such that
    similar texts produce fingerprints with low Hamming distance.

    Use cases:
    - Near-duplicate detection (O(1) comparison per record).
    - Quick rejection before expensive computations.

    Memory: 8 bytes per record (64-bit fingerprint).
    """

    def __init__(self, bits: int = 64) -> None:
        self.bits = bits

    def fingerprint(self, text: str) -> int:
        """
        Compute a SimHash fingerprint.

        Algorithm:
        1. For each token, compute a hash h(token).
        2. For each bit position i:
           - If bit i of h(token) is 1, increment v[i].
           - Otherwise, decrement v[i].
        3. The fingerprint bit i = 1 if v[i] > 0, else 0.

        Returns:
            An integer (fingerprint) of self.bits bits.
        """
        tokens = text.lower().split()
        if not tokens:
            return 0

        v = [0] * self.bits
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8", errors="replace")).hexdigest(), 16)
            for i in range(self.bits):
                v[i] += 1 if (h >> i) & 1 else -1

        result = 0
        for i in range(self.bits):
            if v[i] > 0:
                result |= (1 << i)
        return result

    def hamming_distance(self, fp_a: int, fp_b: int) -> int:
        """Count differing bits between two fingerprints."""
        return bin(fp_a ^ fp_b).count("1")

    def similarity(self, fp_a: int, fp_b: int) -> float:
        """Normalize Hamming distance to [0, 1] similarity score."""
        return 1.0 - self.hamming_distance(fp_a, fp_b) / self.bits
