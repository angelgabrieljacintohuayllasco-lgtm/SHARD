"""
TF-IDF Writer — Posting lists para keyword search sin ML.

Uso:
  python -m shard.index.tfidf_writer --input db/ --out tfidf.shard/ --shards 1000

Genera SHARD DB secundaria con:
  key="término" → value='[{"id": "001", "tfidf": 0.85}, {"id": "023", "tfidf": 0.42}]'
"""

import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List

from shard.storage.shard_writer import ShardWriter

def simple_tokenize(text: str) -> List[str]:
    """Tokeniza lower, sin accents, stems basic (no lib externa)."""
    text = text.lower()
    words = re.findall(r"\b[a-záéíóúüñ]{3,}\b", text)
    return words

def build_tfidf(shard_db_path: str, out_dir: str, num_shards: int = 1000):
    print(f"[TFIDF] Leyendo keys de {shard_db_path}...")
    
    # Recopilar vocab y DF
    df = defaultdict(int)
    corpus_size = 0
    all_records = []
    
    from shard.storage.mmap_reader import MMapReader
    with MMapReader(shard_db_path, num_shards=num_shards) as reader:
        for key in reader.iter_keys():  # asume iter_keys existe o scan
            raw = reader.find(key)
            if raw:
                record = json.loads(raw)
                text = record.get("text", "") + " " + record.get("definition", "")
                tokens = simple_tokenize(text)
                all_records.append((key, tokens))
                unique_tokens = set(tokens)
                for t in unique_tokens:
                    df[t] += 1
                corpus_size += 1
    
    print(f"  Vocab: {len(df)}, Docs: {corpus_size:,}")
    
    # IDF = log(N / DF)
    idf = {term: np.log(corpus_size / (df[term] + 1)) for term in df}
    
    # Build postings
    postings = defaultdict(list)
    for doc_id, tokens in enumerate(all_records):
        key, token_list = all_records[doc_id]
        tf = Counter(token_list)
        doc_len = len(token_list)
        for term, freq in tf.items():
            tfidf = (freq / doc_len) * idf[term]
            postings[term].append({"id": key, "tfidf": float(tfidf)})
    
    # Top 100 por posting (compresión)
    for term in postings:
        postings[term] = sorted(postings[term], key=lambda x: x["tfidf"], reverse=True)[:100]
    
    # Escribir como SHARD
    with ShardWriter(out_dir, num_shards=num_shards) as writer:
        for term, post_list in postings.items():
            writer.write(term, json.dumps(post_list))
    
    print(f"[TFIDF] {len(postings):,} postings escritos en {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser
