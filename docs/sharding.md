# SHARD Sharding Algorithm and Tuning Guide

## How Sharding Works

SHARD distributes records across multiple binary files (shards) using a deterministic hash function. Given a key, the shard ID is computed as:

```
shard_id = FNV1a_64(key.encode("utf-8")) % num_shards
```

This guarantees:
1. The same key always maps to the same shard (deterministic, cross-platform).
2. Records are distributed approximately uniformly across shards.
3. No coordination required between writers (shards can be built in parallel).

## FNV-1a Hash Function

FNV-1a (Fowler–Noll–Vo, variant 1a) is chosen because:

- **Deterministic**: Not subject to Python's `PYTHONHASHSEED` randomization.
- **Fast**: Single pass, no cryptographic overhead. ~1 ns/byte.
- **Good distribution**: Low collision rate for dictionary words and short strings.
- **Cross-language**: Identical results in Python, Rust, Go, C.

```
Algorithm FNV-1a (64-bit):
  h = 0xcbf29ce484222325   (offset basis)
  for each byte b in key:
    h = h XOR b
    h = (h × 0x00000100000001b3) mod 2^64
  return h
```

## Choosing `num_shards`

The optimal number of shards depends on your dataset size and hardware.

### Target rule of thumb

**Aim for 4–16 MB per shard for best performance.**

Too few shards (e.g. 10) → large shard files → long linear scans.
Too many shards (e.g. 100,000) → millions of tiny files → filesystem overhead.

### Recommended values

| Dataset records | Avg record bytes | Recommended shards | Approximate shard size |
|---|---|---|---|
| 10,000 | 512 B | 2 | ~2.5 MB |
| 100,000 | 512 B | 16 | ~3 MB |
| 1,000,000 | 512 B | 128 | ~4 MB |
| 10,000,000 | 512 B | 1,280 | ~4 MB |
| 1,000,000 | 2,000 B | 512 | ~4 MB |

Use `ShardRouter.recommended_shards(estimated_records, avg_record_bytes)` for automatic calculation.

### Power-of-2 recommendation

Use powers of 2 for `num_shards` (16, 32, 64, 128, ..., 4096, ...).

The modulo operation `hash % num_shards` distributes evenly for power-of-2 values when using a good hash function.

## Shard File Naming

Shard files are named: `shard_XXXXXX.bin` where XXXXXX is the zero-padded shard index.

Examples:
```
shard_000000.bin   → shard 0
shard_000384.bin   → shard 384
shard_000999.bin   → shard 999
```

## Multi-Writer (Parallel Build)

Since keys are deterministically assigned to shards, you can build parts of the database in parallel and merge shard files after:

```
Worker 1: writes records 0..250k   → writes shard_000001.bin, shard_000005.bin ...
Worker 2: writes records 250k..500k → same shard files, appended
```

Because `ShardWriter` opens files in append mode (`"ab"`), parallel writers for different key ranges can safely target the same directory, as long as no two workers write the same key.

## Memory Mapping Performance

`MMapReader` uses Python's `mmap.mmap()` with `ACCESS_READ`. The OS manages which pages are in RAM:

- On first access to a shard, the OS pages in the first block (~4 KB).
- Subsequent sequential scans access contiguous pages — very cache-friendly.
- Unaccessed portions of large shard files use zero RAM.
- Pages are evicted under memory pressure, automatically.

This is why SHARD works on 2 GB RAM with 1 TB of data: at any given moment, only the queried pages are in RAM.
