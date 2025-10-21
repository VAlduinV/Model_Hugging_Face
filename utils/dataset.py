
"""
utils/dataset.py

Helpers to read FastText-style datasets used in the user's Lab 2.
Each line is structured like:
__label__1 some text...
__label__2 another text...

We assume:
    __label__1 -> NEGATIVE
    __label__2 -> POSITIVE
If your dataset uses the opposite mapping, pass --invert-labels to the CLI.

This module exposes:
- read_fasttext_file(path, invert_labels=False) -> List[Tuple[str, int]]
- to_dataframe(records) -> pandas.DataFrame with columns ["text", "label"]
- save_json(obj, path)
"""
from __future__ import annotations

import json
from typing import Iterable, List, Tuple
import re
import pandas as pd

LABEL_RE = re.compile(r"^__label__([0-9]+)\s+")

def read_fasttext_file(path: str, invert_labels: bool = False) -> List[Tuple[str, int]]:
    """
    Read a FastText-formatted file into (text, label_id) tuples.
    label_id is 0 for NEGATIVE, 1 for POSITIVE.
    """
    records: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LABEL_RE.match(line)
            if not m:
                # If a line has no label, skip it quietly.
                continue
            label_num = int(m.group(1))
            text = line[m.end():].strip()
            # Default mapping: 1 -> NEGATIVE(0), 2 -> POSITIVE(1)
            if label_num == 1:
                y = 0
            elif label_num == 2:
                y = 1
            else:
                # Unknown labels map to -1; we will filter later.
                y = -1
            if invert_labels and y in (0, 1):
                y = 1 - y
            if y != -1:
                records.append((text, y))
    return records

def to_dataframe(records: List[Tuple[str, int]]) -> "pd.DataFrame":
    """Convert list of (text, label) into a DataFrame."""
    return pd.DataFrame(records, columns=["text", "label"])

def save_json(obj, path: str) -> None:
    """Pretty-save JSON to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
