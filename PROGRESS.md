# Progress log — Vietnamese Hate Speech Detection (ViHSD + PhoBERT)

This file is a handoff doc: if a different AI session or person picks up this project, read this
first. It explains *why* things are structured this way, what's already working, what's still
broken, and what to do next — not just a changelog.

## Goal

Build a CV-worthy NLP project: classify Vietnamese social-media comments as **CLEAN / OFFENSIVE /
HATE**, grounded in actual papers (not just "call PhoBERT and hope"), with a staged/compared
training pipeline run on Kaggle (public, linkable notebook) plus a local demo web app.

Full research rationale (which paper justifies which design decision) is in
`docs/research_notes.md` — read that before changing modeling choices.

## Repo layout

```
data/                        ViHSD train/dev/test CSVs (Kaggle: leduc13/vihsd-data)
notebook/
  vihsd-phobert.ipynb         The Kaggle kernel — staged training pipeline (see below)
  kernel-metadata.json         Kaggle push config (id, GPU/internet flags, dataset_sources)
  log*.txt, output*/           Downloaded run logs from past pushes (debugging artifacts, gitignore-able)
docs/
  research_notes.md            Paper citations + why each design decision was made
backend/
  main.py                      FastAPI inference server
  model/baseline.joblib        TF-IDF+LogReg baseline, trained locally, already working
frontend/
  index.html                   Single-page demo UI (no build step), calls the backend
PROGRESS.md                    This file
README.md                      User-facing summary (setup instructions, CV bullet draft)
```

## Current status (as of last update)

- ✅ **Local demo works end-to-end right now**, no GPU needed:
  - `backend/model/baseline.joblib` — TF-IDF+LogReg, test macro-F1 ≈ 0.629, already trained and saved.
  - `backend/main.py` — FastAPI `/predict` + `/health`, serves the baseline by default
    (`MODEL_BACKEND=baseline`). Swaps to a PhoBERT checkpoint via `MODEL_BACKEND=phobert
    MODEL_DIR=<path>` once one exists.
  - `frontend/index.html` — plain HTML/CSS/JS, calls `API_BASE` (defaults to
    `http://127.0.0.1:8123`). Uses the dataviz-skill status palette (good/warning/critical) for
    CLEAN/OFFENSIVE/HATE since these are severity states, not arbitrary categories.
  - To run: `uvicorn backend.main:app --port 8123`, then open `frontend/index.html` (or serve it
    with `python -m http.server 8080 -d frontend`).

- ✅ **Kaggle notebook — trained successfully, focal-loss bug found and fixed.** Update
  (later session): all 4 stages completed on `hongnhungnguyenthi`. Stage 3 (focal loss) initially
  collapsed (macro-F1 0.207, accuracy 8-12%) due to a `p_t` computation bug (class weight leaking
  into the confidence term via `pt = exp(-weighted_ce)` instead of unweighted CE) — fixed, verified
  with unit-style sanity checks, re-ran: macro-F1 recovered to 0.533 (no longer collapsed). XLM-R
  (Stage 4) is the overall winner at macro-F1 0.625 and is what's saved as `phobert-vihsd-final`
  (a previously-confusing name — it's actually XLM-R, not PhoBERT, since the save-best-stage cell
  picks whichever architecture wins). Full writeup in `docs/research_notes.md`. Also added: an LLM
  (Claude API) feature-engineering experiment (`scripts/llm_feature_extraction.py` +
  `scripts/combine_llm_features.py`) that improves macro-F1 by +0.02, concentrated on the
  OFFENSIVE/HATE boundary. See `README.md`'s Status table for the full metrics.

- ✅ **Ensemble of all 4 models + Claude Sonnet features, validated on real test.csv (later session).**
  Added a probability-dump cell to the notebook (section 11) so `dev`/`test` predictions from all
  4 models get saved as small CSVs without needing to re-download any checkpoint weights. New
  script `scripts/build_ensemble.py` stacks them with a `GradientBoostingClassifier` and evaluates
  once on the real test set: XLM-R alone 0.616 → 4-model ensemble 0.627 (full test) → **0.735** with
  Sonnet-extracted features added, on the OFFENSIVE/HATE-heavy subset (vs. 0.706 without LLM
  features on that same subset). This also closes a gap in the earlier Haiku experiment, which was
  only ever validated via cross-validation on `dev.csv`, never on `test.csv`. Also fixed
  `backend/main.py::_load_phobert` to detect `model_type` from the checkpoint's `config.json` and
  only apply `pyvi` segmentation for actual PhoBERT checkpoints (it previously always segmented,
  which would have silently fed XLM-R text it never saw in training). Full methodology, including
  a `row_index` alignment gotcha in `llm_feature_extraction.py`'s sampling, is in
  `docs/research_notes.md`.

- ⚠️→✅ **Mid-session regression, caught and fixed**: pushing the notebook for the ensemble pass
  above (kernel v9) turned out to still have the *original* buggy `FocalLoss` — the fix from the
  bug write-up below had only ever been applied live in Kaggle's editor, never committed to this
  file. v9's Stage 3 silently collapsed again (macro-F1 0.19, no CLEAN predictions), corrupting the
  PhoBERT-focal probabilities feeding the ensemble above. Caught because the user independently
  re-derived v9's metrics and noticed the collapse. Fixed for real in kernel v10 (`_pt()` now a
  structurally-independent method, plus 2 inline sanity-check assertions that would have caught
  this automatically) — re-verified macro-F1 0.554, no longer collapsed, and the ensemble numbers
  above are from the corrected v10 run.

- 🔲 (superseded below, kept for history) **Kaggle notebook (the real ML work) — in progress, iterating on environment bugs.**
  Kernel: `hongnhungnguyenthi/vihsd-phobert-hate-speech` (was previously also pushed under
  `vuhahodo` — that account's kernel is stale, ignore it; `hongnhungnguyenthi` is the one that
  actually has phone-verified GPU/internet access).

  The notebook (`notebook/vihsd-phobert.ipynb`) is a 4-stage pipeline:
  1. **Stage 1** — TF-IDF + Logistic Regression baseline.
  2. **Stage 2** — PhoBERT-base fine-tuned, word-segmented (`pyvi`), class-weighted cross-entropy.
  3. **Stage 3** — Same PhoBERT setup but with focal loss instead of class weights (compare which
     imbalance-handling mechanism actually wins empirically — see `docs/research_notes.md`).
  4. **Stage 4** — XLM-R-base comparison, GPU-only (guarded by `torch.cuda.is_available()`), fed
     `text_clean` (unsegmented) since XLM-R wasn't pretrained on segmented Vietnamese.
  5. Final cell picks whichever stage scored highest test macro-F1 and saves it to
     `/kaggle/working/phobert-vihsd-final` — that's the path `backend/main.py` expects when you
     point it at a PhoBERT checkpoint.

## Bugs found and fixed, in the order encountered (don't re-discover these)

1. **Dataset not mounting at the hardcoded path.** Fixed by globbing
   `/kaggle/input/**/train.csv` instead of assuming `/kaggle/input/vihsd-data/` (cell `c4`).
2. **Kaggle account not phone-verified** → kernels silently got no internet (pip installs failed)
   and no GPU regardless of `enable_gpu`/`enable_internet` in `kernel-metadata.json`. Not
   fixable from code — required switching to a verified account (`hongnhungnguyenthi`).
3. **GPU assigned was a Tesla P100 (compute capability 6.0 / "sm_60")**, and the environment's
   preinstalled PyTorch build only ships `sm_70+` kernels (PyTorch dropped Pascal/Maxwell around
   the 2.5 release series) → `AcceleratorError: no kernel image is available for execution`.
   Fix: cell `c1` detects compute capability via `nvidia-smi` before importing torch anywhere,
   and if `<7.0`, reinstalls a specific pinned torch version. **This must happen before any
   `import torch` in the notebook** — a CUDA extension can't be swapped out once loaded into a
   running process, so the fix cell has to be first.
4. **First pin attempt (`torch==2.3.1`) broke `transformers`**: `AttributeError: module
   'torch.library' has no attribute 'register_fake'` — that API was only added in torch 2.4.
5. **Bumped to `torch==2.4.1`**: fixed that, but then `torchvision`/`torchaudio` (still pinned to
   the environment's original `torch==2.10.0`) threw `RuntimeError: operator torchvision::nms
   does not exist` on import (mismatched compiled ops). Fix: this notebook has zero vision/audio
   use, so `pip uninstall torchvision torchaudio` instead of chasing matching versions.
6. **`transformers`'s security gate on `torch.load`**: recent `transformers` versions refuse to
   load any checkpoint via `torch.load` unless `torch>=2.6` (a CVE mitigation) — but `torch>=2.6`
   is exactly the version range that dropped `sm_60` support, so no single torch version
   satisfies both constraints. The escape hatch: this restriction **does not apply to
   safetensors-format checkpoints**. Fix: added `use_safetensors=True` to all three
   `AutoModelForSequenceClassification.from_pretrained(...)` calls (Stage 2, 3, 4).
7. **A self-inflicted bug worth knowing about**: cells originally written via the `Write` tool
   (raw JSON, no explicit `"id"` field per cell) get a *display-only* `cell-N` label from
   position at read time. Using `NotebookEdit` with `cell_id="cell-N"` on such a notebook AFTER
   inserting new cells earlier in the file silently edits the wrong cell (positions shifted).
   The notebook was rebuilt from scratch with explicit `"id"` fields on every cell (`c0`, `c1`,
   ...) specifically to make this impossible going forward. **If you add more cells, always
   check that every cell in the file has an explicit `"id"` before relying on `NotebookEdit`
   with a `cell_id` reference — don't reintroduce id-less cells.**

## What to check first if you're picking this up

1. Kernel status: `kaggle kernels status hongnhungnguyenthi/vihsd-phobert-hate-speech`
2. If `ERROR`: `kaggle kernels output hongnhungnguyenthi/vihsd-phobert-hate-speech -p <dir>` and
   read the log (use `PYTHONUTF8=1` env var on Windows or you'll get charmap codec crashes on the
   Vietnamese text in the log — this bit us multiple times).
3. If `COMPLETE`: check the Stage 1–4 macro-F1 numbers in the notebook output, then download the
   saved checkpoint from `/kaggle/working/phobert-vihsd-final` (via `kaggle kernels output` or the
   Kaggle UI's "Output" tab) into somewhere local, and wire up `backend/main.py` with
   `MODEL_BACKEND=phobert MODEL_DIR=<that path>` — the frontend needs zero changes, it just
   reflects whatever `/health` reports as `backend`.

## Not done yet / explicitly deferred

- Multi-seed variance reporting (single seed per stage currently).
- Slang/teencode normalization pass before segmentation (documented as an open question in
  `docs/research_notes.md`, not implemented) — considered and deliberately skipped this round in
  favor of the ensemble + LLM-feature work, which had a clearer expected payoff for less GPU risk.
- No thread-level/context modeling — every prediction is single-comment only, which is the
  documented ceiling on the OFFENSIVE/HATE confusion (see `docs/research_notes.md`).
- The validated 4-model + LLM ensemble (macro-F1 0.737 on the OFFENSIVE/HATE-heavy subset) is not
  wired into `backend/main.py` — serving it live would mean keeping 3 transformer checkpoints
  loaded plus a live Claude API call per request, which is disproportionate for a local demo. It's
  documented as a research result (see `docs/research_notes.md`); the backend still serves a single
  checkpoint via `MODEL_BACKEND=phobert`.
