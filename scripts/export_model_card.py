"""Build model_card.json files (VeriReview contract 4.4 draft) from the result files."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "results"
final = json.loads((res / "final_test_t1.json").read_text(encoding="utf-8"))
lat = json.loads((res / "latency_t1.json").read_text(encoding="utf-8"))
phob = json.loads((res / "final_test_t1_phobert.json").read_text(encoding="utf-8"))
BLOCK_NAMES = ["b1_word_tfidf_pyvi", "b2_char_tfidf_2_5", "b3_surface_stats", "b4_ad_dictionary"]

COMMON = {
    "task": "ViSpamReviews T1: spam (1) vs non-spam (0) from review text only",
    "input_spec": {"type": "string", "description": "review text, Vietnamese; callers must apply the same NFC normalisation as vispam.data before scoring"},
    "output_spec": {"score": "signed margin, larger = more likely spam; score >= 0 predicts spam",
                    "label": "0 non-spam, 1 spam"},
    "train_data": {"name": "ViSpamReviews (CC-BY-NC-4.0), mirror visolex/ViSpamReviews",
                   "split": "published split after exact casefolded de-duplication: 14,181 train / 1,587 dev / 3,971 test",
                   "fit": "train only; test scored once"},
    "limitations": [
        "Source-task model score, not a fake-review probability and not a Y0/Y1/YU label.",
        "Trained on ViSpamReviews (older Vietnamese e-commerce reviews); not validated on self-collected Shopee reviews.",
        "No Golden Test Set metrics yet; no shop, time or reviewer metadata is used.",
        "71/3,971 test rows have cosine >= 0.90 with a train row; baseline macro F1 drops 0.0065 without them.",
        "Weaker than PhoBERT-base-v2 (macro F1 0.8945, paired difference significant, McNemar p < 0.001).",
    ],
    "golden_test_set_metrics": None,
}


def card(key: str, version: str, desc: str) -> dict:
    m, l = final["models"][key], lat[key]
    return {
        "model_version": version, "description": desc, **COMMON,
        "metrics": {"test_macro_f1": m["macro_f1"], "test_macro_f1_ci95": m["ci95"],
                    "test_f1_per_class_nonspam_spam": m["f1_per_class"], "n_test": final["n_test"],
                    "dev_macro_f1": l["dev_macro_f1"]},
        "config": {"blocks_keep_fraction": dict(zip(BLOCK_NAMES, m["keep"])),
                   "block_weights": dict(zip(BLOCK_NAMES, m["weights"])),
                   "classifier": m["classifier"], "C": m["c"], "n_features": m["n_features"]},
        "thresholds": {"decision_margin": 0.0},
        "latency_cpu_ms": {"single_review_median": l["latency_ms_single_median"],
                           "single_review_p95": l["latency_ms_single_p95"],
                           "batch_of_300_mean": l["latency_ms_batch300"],
                           "note": "raw text to prediction, one process, includes pyvi tokenisation"},
        "file": "artifacts/t1_%s.joblib (rebuilt by scripts/measure_latency.py; not committed)" % key.lower(),
        "artifact_mb": l["artifact_mb"],
        "comparison": {"phobert_test_macro_f1": phob["phobert"]["macro_f1"]},
    }


out = ROOT / "results" / "model_cards"
out.mkdir(exist_ok=True)
cards = {"ea": card("EA", "vispam-t1-classical-ea-v1",
                    "Block-selected classical model: knee of multi-objective DE search (24,060 features)"),
         "baseline": card("BASE", "vispam-t1-char-tfidf-lr-v1", "VeriReview baseline: char TF-IDF 2-5 + balanced LR")}
for k, v in cards.items():
    (out / f"t1_{k}_model_card.json").write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
print("written", [p.name for p in out.iterdir()])
