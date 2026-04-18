# SHARD Binary Format Specification — v1

## Overview

Every SHARD database consists of:
- One or more shard files: `shard_XXXXXX.bin`
- Companion Bloom filter files: `shard_XXXXXX.bloom`
- Index files: `index.minhash.bin`, `index.keymap.json`, `index.meta.json`

---

## Shard File Structure

### File Header (8 bytes)

Present at byte offset 0 of every `.bin` file:

```
Offset  Size  Type      Value         Description
------  ----  ----      -----         -----------
0       4     bytes     b"SHRD"       Magic number
4       2     uint16BE  1             Format version
6       2     uint16BE  0             Reserved (must be 0)
```

### Record Layout

Records follow immediately after the 8-byte header, packed sequentially:

```
┌──────────────────────────────────────────────────────────────────┐
│  Field          │  Size     │  Type        │  Description         │
├─────────────────┼───────────┼──────────────┼──────────────────────┤
│  record_length  │  4 bytes  │  uint32 BE   │  Total bytes from    │
│                 │           │              │  key_length to CRC   │
├─────────────────┼───────────┼──────────────┼──────────────────────┤
│  key_length     │  2 bytes  │  uint16 BE   │  Byte length of key  │
├─────────────────┼───────────┼──────────────┼──────────────────────┤
│  key            │  N bytes  │  UTF-8       │  Record key string   │
├─────────────────┼───────────┼──────────────┼──────────────────────┤
│  value_length   │  4 bytes  │  uint32 BE   │  Byte length of val  │
├─────────────────┼───────────┼──────────────┼──────────────────────┤
│  value          │  M bytes  │  UTF-8       │  Record value string │
├─────────────────┼───────────┼──────────────┼──────────────────────┤
│  crc32          │  4 bytes  │  uint32 BE   │  CRC32 of all fields │
│                 │           │              │  from key_length to  │
│                 │           │              │  end of value        │
└──────────────────────────────────────────────────────────────────┘
```

**Total record size**: 4 + 2 + N + 4 + M + 4 = 14 + N + M bytes

**Key constraints**:
- Max key length: 65,535 bytes (uint16 max)
- Max value length: 4,294,967,295 bytes (uint32 max, ~4 GB)
- Encoding: UTF-8 (both key and value)

### Skip-Ahead Algorithm

The `record_length` prefix enables O(1) skip without parsing:

```python
offset = HEADER_SIZE   # 8
while offset < file_size:
    record_length = read_uint32_be(file, offset)
    # process record at offset + 4
    offset += 4 + record_length
```

### CRC32 Integrity

The CRC32 covers all bytes from `key_length` through the end of `value`:

```
CRC32( key_length_bytes || key_bytes || value_length_bytes || value_bytes )
```

On read, SHARD recomputes the CRC and raises `ValueError` on mismatch.

---

## Bloom Filter File Structure (`.bloom`)

```
Offset  Size   Type      Description
------  ----   ----      -----------
0       4      uint32BE  capacity (declared at build time)
4       4      uint32BE  num_bits
8       8      float64BE false_positive_rate
16      ⌈num_bits/8⌉  bytes  bit array
```

---

## MinHash Index File (`index.minhash.bin`)

Flat binary, one row per record:

```
Per-row layout (num_hashes = H):
  [4 bytes: record_id (uint32BE)]
  [4 bytes: shard_id  (uint32BE)]
  [H × 4 bytes: signature (uint32BE array)]

Row size = 8 + H × 4 bytes
```

Default: H=128 → 520 bytes per row → 520 MB for 1M records.

---

## Limits

| Property | Limit |
|---|---|
| Max key length | 65,535 bytes |
| Max value length | ~4 GB |
| Max shards | 4,294,967,295 (uint32) |
| Max records per shard | Unlimited (linear scan) |
| Max total records | Unlimited (disk-bounded) |
| Format version | 1 |
