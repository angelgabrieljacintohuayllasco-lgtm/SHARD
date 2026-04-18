"""
SHARD Build Example
===================
Converts a JSON dataset into a SHARD binary database.

Uses the Spanish dictionary sample from the DASA/SHARD design document
(the same records shown in the original architecture conversation).

After running this script, use query_example.py to search the database.

Run:
    python examples/build_from_json.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shard.storage.shard_writer import ShardWriter
from shard.index.index_builder import IndexBuilder

# ── Demo dataset: Spanish dictionary excerpt ───────────────────────────────────
DEMO_DATA = [
    {"id": "00001", "lemma": "a",            "definition": "Preposición que indica relaciones espaciales, temporales y de destino."},
    {"id": "00002", "lemma": "abaá",         "definition": "Una persona que trabaja como campesino."},
    {"id": "00003", "lemma": "ababillarse",  "definition": "Envolverse con algo."},
    {"id": "00004", "lemma": "ababol",       "definition": "Planta de la familia de las compuestas, conocida por sus flores silvestres."},
    {"id": "00005", "lemma": "abacá",        "definition": "Una fibra natural producida por la planta Musa textilis, usada en cuerdas."},
    {"id": "00006", "lemma": "abacal",       "definition": "Red de cuerdas con una bolsa en el medio utilizada para pescar en ríos."},
    {"id": "00007", "lemma": "abacería",     "definition": "Tienda que vende alimentos y productos relacionados, como especias, vinos y aceites."},
    {"id": "00008", "lemma": "abacial",      "definition": "Relativo a una abadía o al abad que la dirige."},
    {"id": "00009", "lemma": "ábaco",        "definition": "Instrumento de cálculo compuesto de una tabla con barras y cuentas móviles."},
    {"id": "00010", "lemma": "abacora",      "definition": "Especie de pez de agua salada de la familia de los túnidos."},
    {"id": "00011", "lemma": "abacorar",     "definition": "Obtener el favor o la simpatía de alguien mediante halagos."},
    {"id": "00012", "lemma": "abad",         "definition": "Superior de un monasterio o abadía que tiene autoridad sobre los monjes."},
    {"id": "00013", "lemma": "abadejo",      "definition": "Pez comestible de aguas frías, también llamado bacalao."},
    {"id": "00014", "lemma": "abadesa",      "definition": "Superiora de un monasterio femenino o convento de monjas."},
    {"id": "00015", "lemma": "abadía",       "definition": "Monasterio gobernado por un abad o una abadesa, generalmente con iglesia adjunta."},
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_db")
NUM_SHARDS = 16  # Small number for demo (full dataset has few records)
NUM_HASHES = 64


def main() -> None:
    print("SHARD Build Example")
    print("=" * 50)
    print(f"Input   : {len(DEMO_DATA)} records (in-memory demo)")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Shards  : {NUM_SHARDS}")
    print(f"Hashes  : {NUM_HASHES}")
    print()

    start = time.time()

    with ShardWriter(OUTPUT_DIR, num_shards=NUM_SHARDS, bloom_capacity=100) as writer:
        builder = IndexBuilder(OUTPUT_DIR, num_shards=NUM_SHARDS, num_hashes=NUM_HASHES)

        for i, record in enumerate(DEMO_DATA):
            key = record["lemma"]
            value = json.dumps(record, ensure_ascii=False)
            writer.write(key, value)

            text = f"{key} {record['definition']}"
            builder.add(i, key, text)

        builder.build()

    elapsed = time.time() - start

    print(f"Written : {writer.total_written} records")
    print(f"Time    : {elapsed * 1000:.1f} ms")
    print()
    print("Database ready. Run query_example.py to search it.")


if __name__ == "__main__":
    main()
