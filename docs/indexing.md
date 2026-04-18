# SHARD Similarity Index — MinHash

## Overview

The similarity index enables **approximate semantic search**: find records similar to a query even when the exact key is unknown.

This is distinct from SHARD's primary use case (exact lookup). The index is an optional layer built on top of the binary shards.

## Index Files

After calling `IndexBuilder.build()`, three files appear in the database directory:

| File | Size | Contents |
|---|---|---|
| `index.minhash.bin` | N × (8 + H×4) bytes | MinHash signatures |
| `index.keymap.json` | ~30 bytes/record | record_id → key mapping |
| `index.meta.json` | ~100 bytes | Build parameters |

For 1M records at 128 hashes: ~520 MB for the signature matrix.

## How MinHash Works

MinHash estimates Jaccard similarity between two text documents without comparing them directly.

**Step 1: Convert text to a k-shingle set**

```
text = "receta de huevos fritos"
k = 3 (3-word shingles)
shingles = {"receta de huevos", "de huevos fritos"}
```

**Step 2: Apply H hash functions, take the minimum**

For each hash function $h_i$:
$$\text{sig}[i] = \min_{s \in \text{shingles}} h_i(s)$$

This gives a signature array of H integers.

**Step 3: Estimate Jaccard similarity**

$$J(A, B) \approx \frac{|\{i : \text{sig}_A[i] = \text{sig}_B[i]\}|}{H}$$

The more hash positions match, the more similar the documents are.

## RAM Usage

| Records | Hashes | Signature matrix | Key map | Total |
|---|---|---|---|---|
| 10k | 64 | 2.5 MB | 0.3 MB | ~3 MB |
| 100k | 128 | 51 MB | 3 MB | ~54 MB |
| 1M | 128 | 512 MB | 30 MB | ~542 MB |
| 10M | 128 | 5 GB | 300 MB | ~5.3 GB |

For very large datasets (> available RAM), consider:
- Using fewer hash functions (H=32 or H=64) to reduce memory.
- Building a distributed index across multiple machines.
- Using approximate indexing with LSH banding (future feature).

## Tuning `num_hashes`

Fewer hashes = less RAM, lower accuracy.
More hashes = more RAM, higher accuracy.

| num_hashes | RAM per 1M records | Estimated Jaccard error |
|---|---|---|
| 32 | 128 MB | ±0.088 |
| 64 | 256 MB | ±0.062 |
| 128 | 512 MB | ±0.044 |
| 256 | 1,024 MB | ±0.031 |

The error bound follows: $\sigma \approx \sqrt{J(1-J)/H}$

## Building the Index

```python
from shard.index.index_builder import IndexBuilder

builder = IndexBuilder("./mydb", num_shards=1000, num_hashes=128)

for i, record in enumerate(my_records):
    text = f"{record['lemma']} {record['definition']}"
    builder.add(i, record["lemma"], text)

builder.build()  # Writes index files to ./mydb/
```

## Querying the Index

```python
from shard.index.index_reader import IndexReader

reader = IndexReader("./mydb")
reader.load()  # Loads signatures matrix into RAM (~512 MB for 1M records)

results = reader.lookup("planta del campo", top_k=5)
for key, score in results:
    print(f"{score:.3f}  {key}")
```

## Combined Workflow (Similarity → Exact Fetch)

```python
from shard.index.index_reader import IndexReader
from shard.storage.mmap_reader import MMapReader
import json

index = IndexReader("./mydb").load()
reader = MMapReader("./mydb", num_shards=1000)

# Find similar keys
candidates = index.lookup("planta del campo", top_k=3)

# Fetch full records
for key, score in candidates:
    raw = reader.find(key)
    record = json.loads(raw)
    print(f"[{score:.3f}] {record['lemma']}: {record['definition']}")

reader.close()
```
