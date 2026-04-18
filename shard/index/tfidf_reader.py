"""
TF-IDF Reader — Keyword Tier 0, <100ms, <60MB RAM, cualquier CPU.

Merge postings de query terms → BM25 rank → top-k keys.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from shard.storage.mmap_reader
