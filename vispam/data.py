"""Load ViSpamReviews with its published split and remove exact cross-split leakage.

The CSV is CC-BY-NC-4.0 with gated access. Keep it under data/ and never commit it.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://huggingface.co/datasets/visolex/ViSpamReviews/resolve/main/ViSpamReviews.csv"
EXPECTED_SHA256 = "675ba637b15e2521f2014d529ad71b14959aee4d2a51bad841a0b720b82b2690"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ViSpamReviews.csv"
SPLITS = ("train", "dev", "test")
TYPE_NAMES = ("NO-SPAM", "SPAM-1", "SPAM-2", "SPAM-3")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def load_splits(path: str | Path = DEFAULT_PATH, check_hash: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return {split: DataFrame[text, label, spam_type]} and an audit dict.

    label: 0 non-spam, 1 spam (task 1). spam_type: 0..3 (task 2).
    Exact casefolded duplicates are dropped within each split and across splits;
    test rows are kept first, then dev, then train, so test stays as published as possible.
    """
    path = Path(path)
    digest = sha256(path)
    if check_hash and digest != EXPECTED_SHA256:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"type", "Comment", "Label", "SpamLabel"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame["type"].isin(SPLITS)].copy()
    frame["text"] = frame["Comment"].map(normalize)
    frame["label"] = pd.to_numeric(frame["Label"], errors="coerce")
    frame["spam_type"] = pd.to_numeric(frame["SpamLabel"], errors="coerce")
    valid = frame["text"].ne("") & frame["label"].isin((0, 1)) & frame["spam_type"].isin((0, 1, 2, 3))
    frame = frame.loc[valid].copy()
    frame[["label", "spam_type"]] = frame[["label", "spam_type"]].astype(int)
    if not ((frame.label == 0) == (frame.spam_type == 0)).all():
        raise ValueError("Label and SpamLabel disagree on non-spam rows")
    frame["key"] = frame["text"].str.casefold()
    source_counts = {part: int((frame.type == part).sum()) for part in SPLITS}

    seen: set[str] = set()
    splits: dict[str, pd.DataFrame] = {}
    removed: dict[str, int] = {}
    for part in ("test", "dev", "train"):
        subset = frame.loc[frame.type.eq(part)].drop_duplicates("key", keep="first")
        subset = subset.loc[~subset.key.isin(seen)]
        removed[part] = source_counts[part] - len(subset)
        seen.update(subset.key)
        splits[part] = subset[["text", "label", "spam_type"]].reset_index(drop=True)

    audit = {
        "source": SOURCE_URL,
        "sha256": digest,
        "source_splits": source_counts,
        "used_splits": {part: len(splits[part]) for part in SPLITS},
        "removed_exact_duplicates": removed,
        "task1_counts": {part: dict(sorted(Counter(splits[part].label).items())) for part in SPLITS},
        "task2_counts": {part: dict(sorted(Counter(splits[part].spam_type).items())) for part in SPLITS},
    }
    return {part: splits[part] for part in SPLITS}, audit


if __name__ == "__main__":
    import json

    _, info = load_splits()
    print(json.dumps(info, indent=2, default=int))
