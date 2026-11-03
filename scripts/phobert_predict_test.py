"""Score the privately fine-tuned PhoBERT (AI4Life Kaggle run) on our test split, T1.

The model was trained on the same deduplicated train split (14,181 rows); this script only
runs inference so that predictions can be paired with the classical models (McNemar).
Usage: python scripts/phobert_predict_test.py --model <dir with phobert_spam_model>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from underthesea import word_tokenize  # noqa: E402

from vispam.data import load_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "phobert_test_preds_t1.npz")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out.name} exists; delete deliberately to redo.")
    test = load_splits()[0]["test"]
    texts = [word_tokenize(t, format="text") for t in test["text"]]
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).eval()
    torch.set_num_threads(10)
    probs, t0 = [], time.time()
    for i in range(0, len(texts), a.batch_size):
        enc = tok(texts[i:i + a.batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy())
        if (i // a.batch_size) % 10 == 0:
            print(f"{i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    prob = np.concatenate(probs)
    wall = time.time() - t0
    np.savez_compressed(a.out, y=test["label"].to_numpy(), pred=(prob >= 0.5).astype(int), prob=prob,
                        ms_per_review_batched=wall / len(texts) * 1000)
    print(f"done in {wall:.0f}s, {wall / len(texts) * 1000:.1f} ms/review (batched, CPU)")


if __name__ == "__main__":
    main()
