# Contributing to SHARD

Thank you for your interest in SHARD!

## Ways to Contribute

- **Bug reports** — Open an issue with a minimal reproduction
- **Performance improvements** — Better scan algorithms, compression strategies
- **New hash functions** — Alternative to FNV1a with better distribution
- **Language bindings** — Rust, Go, or C readers for cross-language use
- **Compression** — Integrate LZ4/zstd for value compression
- **Benchmarks** — Compare SHARD against SQLite, LMDB, RocksDB

## Development Setup

```bash
git clone https://github.com/YOUR_ORG/shard.git
cd shard
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- PEP 8 compliance
- Type hints on all public functions
- Zero external dependencies for `shard/core/` and `shard/storage/`
- `numpy` is allowed only in `shard/index/` and `shard/core/hasher.py`

## The SHARD Contract

All contributions must preserve:

1. **Determinism:** `ShardRouter.get_shard(key)` must always return the same shard for the same key, across platforms and Python versions.
2. **Zero false negatives:** `BloomFilter.contains(key)` must return `True` for every key that was added.
3. **CRC integrity:** Every record written by `ShardWriter` must pass CRC validation when read by `MMapReader`.
4. **No full-file loads:** `MMapReader` must never load an entire shard file into Python memory — only use `mmap` page access.

## Pull Request Process

1. Fork and branch: `git checkout -b feature/my-improvement`
2. Write tests that cover your change
3. Ensure `pytest tests/ -v` passes
4. Open a pull request with benchmarks if the change affects performance
