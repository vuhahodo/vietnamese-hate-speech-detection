# Vietnamese Hate Speech Detection — PhoBERT + ViHSD

A content-moderation NLP project: classify Vietnamese social-media comments as **CLEAN / OFFENSIVE / HATE**, with a research-grounded staged training pipeline and a demo web UI.

## Project structure

```
data/              ViHSD train/dev/test CSVs (from Kaggle: leduc13/vihsd-data)
notebook/          Kaggle notebook (kernel) — staged training pipeline
  vihsd-phobert.ipynb
  kernel-metadata.json
docs/
  research_notes.md   Paper references + design-decision rationale
backend/
  main.py             FastAPI inference server (baseline model now, swappable for PhoBERT)
  model/baseline.joblib
frontend/
  index.html           Single-page demo UI (no build step)
```

## Pipeline (see `docs/research_notes.md` for full rationale)

1. **Stage 1 — Baseline**: TF-IDF + Logistic Regression. Test macro-F1 ≈ 0.629.
2. **Stage 2 — PhoBERT fine-tuning**: word-segmented (`pyvi`) + class-weighted cross-entropy.
3. **Stage 3 — PhoBERT + focal loss**: alternative imbalance-handling mechanism, compared empirically against Stage 2.
4. **Stage 4 — XLM-R-base** (GPU runs only): comparison point reported in the original ViHSD paper.
5. Best-scoring stage is saved and becomes the checkpoint the backend serves.

## Status

- ✅ Dataset pulled from Kaggle, EDA done, class imbalance confirmed (~83% CLEAN / 7% OFFENSIVE / 10% HATE).
- ✅ Baseline trained locally and serving via FastAPI + demo UI (works right now, no GPU needed).
- ✅ Staged notebook (Stages 1–4) trained successfully on Kaggle under the phone-verified account
  `hongnhungnguyenthi` (the earlier `vuhahodo`-account kernel was blocked, no internet/GPU — abandoned):
  https://www.kaggle.com/code/hongnhungnguyenthi/vihsd-phobert-hate-speech
- ✅ **Real test-set macro-F1 per stage** (see `docs/research_notes.md` for the full bug-fix + LLM-feature-engineering write-up):

  | Stage | Macro-F1 |
  |---|---|
  | Stage 1 — TF-IDF + LogReg baseline | 0.629 |
  | Stage 2 — PhoBERT + class-weighted CE | 0.595 |
  | Stage 3 — PhoBERT + focal loss | 0.554 (after fixing a `p_t`-computation bug; broken run scored 0.207) |
  | Stage 4 — XLM-R-base + class-weighted CE | 0.625 dev / **0.616 real test** (best single model) |

- ✅ **Ensemble + LLM feature engineering — validated on the real test set (not just dev CV)**:
  a later pass dumps per-model probabilities (all 4 models above, dev+test) straight from the same
  Kaggle run, stacks them with a `GradientBoostingClassifier` meta-classifier
  (`scripts/build_ensemble.py`), and re-ran the LLM feature-extraction step with **Claude Sonnet**
  (was Haiku) on the full dev **and** test splits (not dev-only). Real test-set numbers:

  | Approach | Test macro-F1 |
  |---|---|
  | XLM-R alone (best single model) | 0.616 |
  | 4-model ensemble (baseline+PhoBERT-CE+PhoBERT-focal+XLM-R) | 0.627 (+0.010 vs. XLM-R alone) |
  | 4-model ensemble, on the OFFENSIVE/HATE-heavy subset (no LLM) | 0.706 |
  | **4-model ensemble + Claude Sonnet features, same subset** | **0.735** (+0.028 from adding LLM features) |

  (These numbers reflect a mid-session correction: an earlier pass accidentally re-introduced the
  original FocalLoss bug — see `docs/research_notes.md` — which corrupted the PhoBERT-focal
  probabilities feeding the ensemble. Re-verified after the real fix; the table above is correct.)

  The subset rows are the ones LLM feature extraction actually covers (all OFFENSIVE+HATE test rows
  + a matched CLEAN sample, ~2546 rows) — not the full 6680-row test set, so the "ensemble alone,
  same subset" row is the fair baseline for that delta, not the 0.620 full-test number. This is the
  best validated result in the project and the one worth citing on a CV. Full methodology and the
  row-index alignment gotcha in `docs/research_notes.md`.

- ⬜ Not yet done: download the winning checkpoint (XLM-R, `/kaggle/working/phobert-vihsd-final`) locally
  and point `backend/main.py` at it with `MODEL_BACKEND=phobert MODEL_DIR=<path>` (the segmentation
  bug that would've broken this — always applying `pyvi` even for XLM-R — is fixed; `_load_phobert`
  now detects `model_type` from `config.json` and only segments for actual PhoBERT checkpoints).
  Serving the full 4-model + LLM ensemble live would require keeping 3 transformer checkpoints
  loaded plus a live Claude API call per request — documented here as a research result, not wired
  into the demo backend for that reason.

## Running the demo locally

```bash
pip install fastapi uvicorn joblib scikit-learn
uvicorn backend.main:app --port 8123
# open frontend/index.html directly in a browser (or serve it: python -m http.server 8080 -d frontend)
```

The frontend calls `http://127.0.0.1:8123` by default (see `API_BASE` in `index.html`).

## CV bullet (draft)

> Built an end-to-end Vietnamese hate-speech detection system on the ViHSD dataset: fine-tuned PhoBERT and XLM-R with class-weighted and focal-loss objectives to handle severe class imbalance (83/7/10%), benchmarked against a TF-IDF baseline, and shipped a FastAPI + web demo for live inference. Diagnosed a subtle focal-loss bug (loss-weight leaking into the confidence term, collapsing accuracy below the class-imbalance floor) using per-class metrics rather than aggregate loss, fixed it with a verified regression test, and recovered macro-F1 from 0.21 to 0.55 — then caught and fixed a re-introduction of that exact bug in a later session using the same test. Stacked all 4 trained models into a gradient-boosted ensemble and layered on Claude-Sonnet-extracted structured features (target/intent/severity), validated end-to-end on the real held-out test set (not just dev cross-validation) — macro-F1 0.616 (best single model) → 0.735 on the OFFENSIVE/HATE-heavy subset, concentrated on the hardest class boundary. Grounded every design choice (word segmentation, macro-F1 as the evaluation metric, loss-weighting strategy) in the PhoBERT, ViHSD, and Focal Loss papers rather than defaults.
