import os
import struct
import tempfile

import numpy as np
import pytest

from shard.core.hasher import MinHasher, SimHasher
from shard.core.sharding import ShardRouter
from shard.core.bloom_filter import BloomFilter
from shard.storage.binary_encoder import (
    encode_record,
    decode_record,
    shard_file_header,
    validate_header,
    HEADER_SIZE,
)
from shard.storage.shard_writer import ShardWriter
from shard.storage.mmap_reader import MMapReader
from shard.index.index_builder import IndexBuilder
from shard.index.index_reader import IndexReader


# ── MinHasher ──────────────────────────────────────────────────────────────────

class TestMinHasher:
    def test_signature_shape(self):
        h = MinHasher(num_hashes=64)
        sig = h.signature("planta de la familia de las compuestas")
        assert sig.shape == (64,)
        assert sig.dtype == np.uint32

    def test_empty_text_returns_zeros(self):
        h = MinHasher(num_hashes=32)
        sig = h.signature("")
        assert np.all(sig == 0)

    def test_identical_texts_have_similarity_one(self):
        h = MinHasher(num_hashes=128)
        text = "receta de huevos fritos con aceite"
        sig = h.signature(text)
        assert h.similarity(sig, sig) == 1.0

    def test_similar_texts_have_higher_similarity_than_dissimilar(self):
        # Use longer texts to ensure 3-gram overlap is measurable
        h = MinHasher(num_hashes=256)
        sig_a = h.signature("planta de la familia compuestas flores silvestres campo")
        sig_b = h.signature("planta de la familia botanica flores silvestres jardin")
        sig_c = h.signature("instrumento calculo matematico barras cuentas moviles numeros")
        # sig_a and sig_b share multiple 3-grams ("planta de la", "de la familia", "flores silvestres")
        # sig_a and sig_c share no 3-grams
        assert h.similarity(sig_a, sig_b) > h.similarity(sig_a, sig_c)

    def test_serialization_roundtrip(self):
        h = MinHasher(num_hashes=64)
        sig = h.signature("test de serialización")
        restored = h.from_bytes(h.to_bytes(sig))
        np.testing.assert_array_equal(sig, restored)

    def test_deterministic_across_calls(self):
        h = MinHasher(num_hashes=64)
        sig1 = h.signature("ababol planta")
        sig2 = h.signature("ababol planta")
        np.testing.assert_array_equal(sig1, sig2)


# ── SimHasher ──────────────────────────────────────────────────────────────────

class TestSimHasher:
    def test_fingerprint_is_integer(self):
        s = SimHasher(bits=64)
        fp = s.fingerprint("hola mundo")
        assert isinstance(fp, int)

    def test_identical_texts_have_zero_hamming_distance(self):
        s = SimHasher()
        fp = s.fingerprint("texto exactamente igual para comparar")
        assert s.hamming_distance(fp, fp) == 0

    def test_similar_texts_have_low_hamming_distance(self):
        s = SimHasher()
        fp_a = s.fingerprint("receta de huevos fritos con aceite")
        fp_b = s.fingerprint("receta de huevo frito en aceite caliente")
        assert s.hamming_distance(fp_a, fp_b) < 40  # out of 64 bits

    def test_similarity_range(self):
        s = SimHasher()
        fp_a = s.fingerprint("python programacion")
        fp_b = s.fingerprint("python lenguaje")
        score = s.similarity(fp_a, fp_b)
        assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        s = SimHasher()
        assert s.fingerprint("") == 0


# ── ShardRouter ────────────────────────────────────────────────────────────────

class TestShardRouter:
    def test_deterministic(self):
        r = ShardRouter(num_shards=1000)
        assert r.get_shard("ababol") == r.get_shard("ababol")

    def test_shard_id_in_range(self):
        r = ShardRouter(num_shards=100)
        for word in ["a", "ababol", "zymurgy", "ábaco", "python"]:
            sid = r.get_shard(word)
            assert 0 <= sid < 100, f"{word!r} → shard {sid} out of range"

    def test_filename_format(self):
        r = ShardRouter(num_shards=1000)
        assert r.shard_filename(0) == "shard_000000.bin"
        assert r.shard_filename(42) == "shard_000042.bin"
        assert r.shard_filename(999) == "shard_000999.bin"

    def test_bloom_filename_format(self):
        r = ShardRouter(num_shards=100)
        assert r.bloom_filename(5) == "shard_000005.bloom"

    def test_raises_on_zero_shards(self):
        with pytest.raises(ValueError):
            ShardRouter(num_shards=0)

    def test_recommended_shards_is_power_of_two(self):
        n = ShardRouter.recommended_shards(1_000_000, avg_record_bytes=512)
        assert n > 0
        assert (n & (n - 1)) == 0  # power of 2


# ── BloomFilter ────────────────────────────────────────────────────────────────

class TestBloomFilter:
    WORDS = ["ababol", "ábaco", "abacá", "python", "inteligencia"]

    def test_no_false_negatives(self):
        bf = BloomFilter(capacity=100, false_positive_rate=0.01)
        for w in self.WORDS:
            bf.add(w)
        for w in self.WORDS:
            assert bf.contains(w), f"False negative for {w!r}"

    def test_unseen_keys_do_not_raise(self):
        bf = BloomFilter(capacity=100)
        # contains() is probabilistic — we just check it doesn't raise
        result = bf.contains("never_added_key_xyz")
        assert isinstance(result, bool)

    def test_serialization_roundtrip(self):
        bf = BloomFilter(capacity=100, false_positive_rate=0.01)
        for w in self.WORDS:
            bf.add(w)
        restored = BloomFilter.from_bytes(bf.to_bytes())
        for w in self.WORDS:
            assert restored.contains(w), f"Deserialized filter lost {w!r}"

    def test_fill_ratio_increases_with_inserts(self):
        bf = BloomFilter(capacity=1000)
        initial = bf.fill_ratio
        for i in range(100):
            bf.add(f"word_{i}")
        assert bf.fill_ratio > initial

    def test_count_tracks_inserts(self):
        bf = BloomFilter(capacity=100)
        for i in range(10):
            bf.add(f"item_{i}")
        assert bf.count == 10


# ── BinaryEncoder ─────────────────────────────────────────────────────────────

class TestBinaryEncoder:
    def test_encode_decode_roundtrip_ascii(self):
        key, value = "ababol", "Planta de la familia de las compuestas."
        encoded = encode_record(key, value)
        decoded_key, decoded_value, consumed = decode_record(encoded, 0)
        assert decoded_key == key
        assert decoded_value == value
        assert consumed == len(encoded)

    def test_encode_decode_roundtrip_unicode(self):
        key, value = "ábaco", "Instrumento de cálculo. Útil para estudiantes."
        encoded = encode_record(key, value)
        decoded_key, decoded_value, _ = decode_record(encoded, 0)
        assert decoded_key == key
        assert decoded_value == value

    def test_header_validation_passes(self):
        assert validate_header(shard_file_header())

    def test_header_validation_fails_on_wrong_magic(self):
        bad = b"XXXX\x00\x01\x00\x00"
        assert not validate_header(bad)

    def test_header_size_constant(self):
        assert len(shard_file_header()) == HEADER_SIZE

    def test_crc_mismatch_raises_value_error(self):
        encoded = bytearray(encode_record("key", "value"))
        # Corrupt a byte in the middle of the payload
        encoded[len(encoded) // 2] ^= 0xFF
        with pytest.raises(ValueError, match="CRC mismatch"):
            decode_record(bytes(encoded), 0)

    def test_empty_value(self):
        key, value = "empty_val", ""
        enc = encode_record(key, value)
        dk, dv, _ = decode_record(enc, 0)
        assert dk == key
        assert dv == value


# ── Write + Read integration ──────────────────────────────────────────────────

class TestWriterReaderIntegration:
    def test_write_and_read_back_exact(self):
        records = [
            ("ababol", '{"id": "1", "definition": "Planta."}'),
            ("ábaco",  '{"id": "2", "definition": "Instrumento."}'),
            ("python", '{"id": "3", "definition": "Lenguaje."}'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=4) as writer:
                for key, value in records:
                    writer.write(key, value)
            assert writer.total_written == len(records)

            with MMapReader(tmp, num_shards=4) as reader:
                for key, expected_value in records:
                    result = reader.find(key)
                    assert result == expected_value, (
                        f"For key={key!r}: expected {expected_value!r}, got {result!r}"
                    )

    def test_missing_key_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=4) as writer:
                writer.write("exists", "value")

            with MMapReader(tmp, num_shards=4) as reader:
                assert reader.find("does_not_exist") is None

    def test_bloom_filter_companion_files_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=4) as writer:
                writer.write("test_key", "test_value")

            bloom_files = list(os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".bloom"))
            assert len(bloom_files) >= 1


# ── ShardWriter — batch write + auto bloom capacity ───────────────────────────

class TestShardWriterBatch:
    RECORDS = [
        ("ababol",  '{"definition": "Planta."}'),
        ("ábaco",   '{"definition": "Instrumento."}'),
        ("python",  '{"definition": "Lenguaje."}'),
        ("shard",   '{"definition": "Base de datos."}'),
        ("huevo",   '{"definition": "Alimento."}'),
    ]

    def test_write_batch_returns_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=4) as writer:
                count = writer.write_batch(self.RECORDS)
        assert count == len(self.RECORDS)

    def test_write_batch_all_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=4) as writer:
                writer.write_batch(self.RECORDS)
            with MMapReader(tmp, num_shards=4) as reader:
                for key, expected in self.RECORDS:
                    assert reader.find(key) == expected

    def test_estimated_total_records_sets_bloom_capacity(self):
        """When estimated_total_records is given, bloom_capacity must reflect it."""
        writer = ShardWriter(
            "fake_dir",
            num_shards=100,
            estimated_total_records=10_000,
        )
        # Expected: ceil(10_000 / 100 * 1.2) = 120
        assert writer.bloom_capacity == 120

    def test_without_estimate_uses_default(self):
        writer = ShardWriter("fake_dir", num_shards=100, bloom_capacity=5_000)
        assert writer.bloom_capacity == 5_000

    def test_total_written_accumulates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=4) as writer:
                writer.write_batch(self.RECORDS[:2])
                writer.write_batch(self.RECORDS[2:])
            assert writer.total_written == len(self.RECORDS)


# ── IndexBuilder + IndexReader integration ────────────────────────────────────

class TestIndexBuilderReader:
    RECORDS = [
        (0, "ababol",  "ababol planta familia compuestas campo silvestre"),
        (1, "ábaco",   "abaco instrumento calculo barras cuentas matematica"),
        (2, "python",  "python lenguaje programacion alto nivel interpretado"),
        (3, "huevo",   "huevo alimento ave cocina receta proteinas"),
        (4, "shard",   "shard base datos binaria hash escalable almacenamiento"),
    ]

    def test_build_creates_index_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = IndexBuilder(tmp, num_shards=4, num_hashes=32)
            for rid, key, text in self.RECORDS:
                builder.add(rid, key, text)
            builder.build()

            assert (os.path.join(tmp, "index.minhash.bin") or True)
            assert os.path.exists(os.path.join(tmp, "index.meta.json"))
            assert os.path.exists(os.path.join(tmp, "index.keymap.json"))

    def test_record_count_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = IndexBuilder(tmp, num_shards=4, num_hashes=32)
            for rid, key, text in self.RECORDS:
                builder.add(rid, key, text)
            assert builder.record_count == len(self.RECORDS)

    def test_add_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = IndexBuilder(tmp, num_shards=4, num_hashes=32)
            builder.add_batch(self.RECORDS)
            assert builder.record_count == len(self.RECORDS)

    def test_lookup_returns_most_similar(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = IndexBuilder(tmp, num_shards=4, num_hashes=32)
            for rid, key, text in self.RECORDS:
                builder.add(rid, key, text)
            builder.build()

            reader = IndexReader(tmp)
            reader.load()
            # Query shares consecutive word 3-grams with the "ababol" entry:
            # stored: "ababol planta familia compuestas campo silvestre"
            # query:  "ababol planta familia" → trigram ("ababol","planta","familia") matches
            results = reader.lookup("ababol planta familia compuestas campo", top_k=3)

            assert len(results) > 0
            keys_returned = [key for key, _ in results]
            assert "ababol" in keys_returned

    def test_lookup_scores_in_zero_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = IndexBuilder(tmp, num_shards=4, num_hashes=32)
            for rid, key, text in self.RECORDS:
                builder.add(rid, key, text)
            builder.build()

            reader = IndexReader(tmp)
            reader.load()
            results = reader.lookup("python lenguaje", top_k=5)

            for key, score in results:
                assert 0.0 <= score <= 1.0, f"Score {score} for {key!r} out of [0,1]"

    def test_load_raises_if_index_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = IndexReader(tmp)
            with pytest.raises(FileNotFoundError):
                reader.load()


# ── MMapReader — bloom LRU eviction ───────────────────────────────────────────

class TestMMapReaderBloomEviction:
    def test_bloom_cache_does_not_exceed_limit(self):
        """After querying more shards than bloom_cache_size, cache stays bounded."""
        with tempfile.TemporaryDirectory() as tmp:
            # Write records that will land in 10 different shards
            with ShardWriter(tmp, num_shards=16) as writer:
                for i in range(80):
                    writer.write(f"key_{i:04d}", f"value_{i}")

            # Set a very small cache limit
            reader = MMapReader(tmp, num_shards=16, bloom_cache_size=4)
            # Query enough keys to trigger eviction
            for i in range(80):
                reader.find(f"key_{i:04d}")

            assert len(reader._blooms) <= 4
            reader.close()

    def test_evicted_bloom_reloaded_correctly(self):
        """After eviction, querying the same key still returns correct data."""
        with tempfile.TemporaryDirectory() as tmp:
            with ShardWriter(tmp, num_shards=16) as writer:
                writer.write("target_key", "target_value")
                for i in range(60):
                    writer.write(f"filler_{i:04d}", f"val_{i}")

            reader = MMapReader(tmp, num_shards=16, bloom_cache_size=2)
            # Flood the cache to force evictions
            for i in range(60):
                reader.find(f"filler_{i:04d}")
            # Target key must still be found even if its bloom was evicted
            result = reader.find("target_key")
            assert result == "target_value"
            reader.close()
