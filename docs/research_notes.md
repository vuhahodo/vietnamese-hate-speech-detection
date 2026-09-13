# Research notes — Vietnamese Hate Speech Detection

## Core references

1. **Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese.** Findings of EMNLP 2020.
   - Two monolingual RoBERTa-base/large models pretrained on ~20GB of word-segmented Vietnamese text.
   - Key implication for this project: PhoBERT's pretraining corpus was segmented with VnCoreNLP/RDRsegmenter (multi-syllable words joined by `_`). Feeding it raw, unsegmented text at fine-tuning time is a train/inference-time mismatch with the pretraining distribution and measurably hurts subword tokenization quality — this is why the pipeline runs a word-segmentation step (`pyvi.ViTokenizer`) before tokenizing.

2. **Luu, S. T., Nguyen, K. V., & Nguyen, N. L. T. (2021). A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts (ViHSD).** ACIIDS 2021.
   - Introduces the ViHSD corpus used here: ~33k Facebook/YouTube comments, 3-way labeled CLEAN / OFFENSIVE / HATE.
   - Reports the same imbalance we see (~83/7/10%) and the same dominant confusion: OFFENSIVE vs HATE, because the boundary is pragmatic (target + intent), not lexical — a comment can contain profanity without targeting a protected group (OFFENSIVE) or target one without profanity (HATE, e.g. dehumanizing language).
   - Their own baselines (TF-IDF+SVM/LogReg, CNN/LSTM/BiGRU) land around macro-F1 0.5–0.6; PhoBERT/XLM-R fine-tuning is reported as their best-performing setup, which motivates using it here rather than a from-scratch neural model.

3. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection.** ICCV 2017.
   - Originally for object detection's foreground/background imbalance, focal loss down-weights easy, already-confident examples (`(1-p_t)^γ` factor) so the loss isn't dominated by the ~83% easy CLEAN examples.
   - Used here as an alternative to plain class-weighted cross-entropy (Stage 3 in the notebook) — the two handle imbalance differently: class weights fix the *loss scale* per class, focal loss fixes *which examples* dominate gradient regardless of class. Worth comparing empirically rather than assuming one wins.

4. **Waseem, Z., & Hovy, D. (2016). Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter.** NAACL SRW.
   - Establishes the now-standard framing that hate speech is a distinct, narrower category than generic offense/toxicity, and that annotator agreement on the boundary is inherently harder than on "is this toxic at all" — relevant context for why OFFENSIVE/HATE confusion is expected to be the main error mode here, not a pipeline bug.

## Design decisions this project makes, and why

| Decision | Grounded in | Alternative considered |
|---|---|---|
| Word-segment with `pyvi` before PhoBERT tokenization | PhoBERT paper's pretraining setup | Skip segmentation (simpler, but degrades subword quality) |
| Report macro-F1 as primary metric, not accuracy | ViHSD paper's own evaluation protocol; ~83% majority class makes accuracy misleading | Weighted-F1 (masks minority-class failures) |
| Class-weighted CE (Stage 2) vs Focal loss (Stage 3) as separate stages | Lin et al. — different imbalance-handling mechanisms, empirically compare rather than pick by default | Naive oversampling only |
| PhoBERT-base vs XLM-R-base comparison | ViHSD paper reports both as strong baselines for Vietnamese | Multilingual mBERT (weaker on Vietnamese per PhoBERT paper's own comparisons) |
| Qualitative error analysis on OFFENSIVE↔HATE pairs specifically | Waseem & Hovy + ViHSD's reported confusion pattern | Generic top-K misclassification dump |

## Bug found and fixed: Focal loss `p_t` computed from weighted CE, not raw CE

The first successful Kaggle run of Stage 3 (focal loss) collapsed: eval_accuracy 8–12% across
all 4 epochs (worse than always predicting CLEAN, worse than always predicting HATE), while
eval_loss was the *lowest* of the three stages — a case where aggregate loss actively misleads,
and only per-class/accuracy inspection reveals a real bug.

Root cause: `pt = torch.exp(-ce)` where `ce` came from
`F.cross_entropy(logits, labels, weight=class_weights, reduction='none')`. PyTorch applies
`weight` directly to the loss value (`ce_i = weight[y_i] * -log p_i`), so
`pt = exp(-ce_i) = p_i ** weight[y_i]` — not the model's true confidence `p_i`. For the
majority class CLEAN (`weight≈0.40`), this pushes `pt` toward 1 for already-easy examples,
so `(1-pt)^gamma` collapses toward 0 and gradient from ~83% of the training data vanishes
almost entirely; for the minority classes (`weight≈3–5`), the opposite distortion happens.
Verified numerically (toy batch, `p_true=0.9`): the buggy formula under-weights CLEAN's loss to
~17% of correct and over-weights OFFENSIVE/HATE's loss by 8–17x relative to a correct
implementation — a large, systematic double distortion, not a rounding error.

Fix: compute `pt` from an **unweighted** per-sample CE (`weight=None`), independent of the
class-weight (`alpha`) term, matching the literature formula
`FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)` where `alpha_t` and `p_t` are independent factors.
Validated with unit-style sanity assertions before the full run (see the notebook cell right
after `FocalLoss`'s definition): `gamma=0` must reduce exactly to `CrossEntropyLoss`, and `p_t`
must be identical whether or not `weight` is passed.

Result after the fix (same 4-epoch schedule, same data): eval_accuracy back to a normal 76–82%
range (matching the CE baseline's range), eval_macro_f1 climbing 0.30 → 0.53 across epochs —
no longer collapsed, though it landed slightly below the CE baseline (0.595) and XLM-R (0.625)
in this run. That's a legitimate empirical result, not a remaining bug: focal loss doesn't
universally beat a well-tuned class-weighted CE, especially without further gamma/alpha tuning
(see "Open questions" below for a gamma sweep as a natural follow-up).

## LLM-based feature engineering experiment (Claude API)

Used Claude (Haiku) as a feature-engineering aid, not a replacement classifier: for each comment
in a stratified sample (all OFFENSIVE+HATE rows + a matched random sample of CLEAN, ~1080 rows
of `dev.csv`), asked it to extract structured pragmatic signals a pure text classifier tends to
miss — whether the comment targets a specific individual/group, whether it contains an explicit
threat or dehumanizing language, a severity score, and an independent suggested label — aimed
directly at the OFFENSIVE-vs-HATE boundary this project's own error analysis (and the ViHSD
paper) identifies as the dominant confusion.

Combined these features with the existing baseline model's per-class probabilities via a small
secondary classifier (gradient boosting, 5-fold CV on the same subset) and compared macro-F1
with vs. without the LLM features:

| | Macro-F1 |
|---|---|
| Baseline model alone (argmax) | 0.653 |
| Re-ranker on model probs only | 0.646 |
| **Re-ranker on model probs + LLM features** | **0.666** (+0.020 vs. without) |
| LLM suggested label alone (no model, reference) | 0.458 |

The gain concentrates exactly where hypothesized: OFFENSIVE F1 improved most (0.444→0.483),
with a smaller HATE gain (0.649→0.664) and negligible CLEAN change (0.846→0.850) — evidence the
LLM-derived target/intent signal helps specifically at the hardest boundary, not just adding
noise that happens to average out favorably. (Scripts: `scripts/llm_feature_extraction.py`,
`scripts/combine_llm_features.py`; raw extractions cached in `data/llm_features_dev.jsonl`.)

Incidental finding worth flagging for anyone reusing this prompt: Haiku consistently misspelled
the JSON key `explicit_threat_or_dehumanizing` as `explicit_threat_or_dehumatizing` (dropped the
"n") in 1074/1084 responses despite the prompt spelling it correctly — the combining script
coalesces both spellings rather than silently dropping those rows to NaN.

## Postscript: the FocalLoss bug was re-introduced, then caught and fixed for real

The write-up above documents the *original* discovery of the FocalLoss bug and its fix, from an
earlier session. In this session, an ensemble-building pass pushed the notebook file that's
actually tracked in this repo (`NLP/notebook/vihsd-phobert.ipynb`) to Kaggle — and it turned out
that file still had the **original buggy** `FocalLoss` (`p_t` computed from the *weighted* CE, not
an independent unweighted one). The documented fix had apparently only ever been applied ad-hoc in
Kaggle's web editor in the earlier session, never committed back to this file. The resulting run
(kernel v9) silently reproduced the exact original collapse (Stage 3: macro-F1 0.19, accuracy
10.7%, no CLEAN predictions at all) — caught because the user independently re-derived the test
metrics from that run and noticed the collapse, not because any tooling flagged it.

Fixed for real this time, plus two additions specifically to prevent a third recurrence:
1. `FocalLoss._pt()` is now a separate method computing `p_t` from an **unweighted** CE, so the
   independence from `self.weight` is structural, not just a comment.
2. Two sanity-check assertions run inline right after the class definition, every time the
   notebook executes: `FocalLoss(gamma=0)` must equal plain weighted `CrossEntropyLoss` exactly,
   and `p_t` must be bit-identical whether or not `weight` is set. These would have caught both the
   original bug and its re-introduction automatically, without needing a human to notice a
   suspicious accuracy number.

Re-verified (kernel v10, same 4-epoch schedule, same data): eval_accuracy back to normal range,
**test macro-F1 = 0.554** — consistent with the originally-reported "~0.53, no longer collapsed"
finding, still below the CE baseline (0.595) and XLM-R (0.625), which remains a legitimate
empirical result (see the "Result after the fix" discussion above), not a remaining bug.

## Ensemble + upgraded LLM feature engineering (Claude Sonnet), validated on the real test set

The Haiku experiment above had two open gaps: (1) it only ever used the sklearn baseline's
probabilities, not the 3 trained transformers, and (2) it was only ever validated via 5-fold CV
on `dev.csv` — never on the actual held-out `test.csv`. This pass closes both.

**Getting per-model probabilities out of the notebook.** None of the 4 trained models had their
predicted probabilities persisted anywhere — only the single "best" checkpoint gets saved, and
`test_logits`/`focal_logits`/`xlmr_test_output` only exist in Kaggle-session memory. Rather than
re-running training in a separate session (expensive, another chance to hit the GPU/torch version
issues documented above), a new cell was appended to the same notebook run (`vihsd-phobert.ipynb`,
section 11) that reuses the still-in-memory `trainer`/`trainer_focal`/`trainer_xlmr`/`clf` objects
to run `predict`/`predict_proba` on **both** `dev_ds` and `test_ds`, and dumps compact
`probs_{model}_{split}.csv` files (3 probability columns + label + text, no weights) to
`/kaggle/working/`. This is the only place actual disk I/O for ensembling happens — everything
else runs downstream, locally, on the small CSVs.

**Ensemble.** `scripts/build_ensemble.py` stacks `[baseline_probs, phobert_ce_probs,
phobert_focal_probs, xlmr_probs]` (12 features) into a `GradientBoostingClassifier`
(`n_estimators=100, max_depth=3`, same family used in the earlier Haiku experiment for
consistency), trained on the full `dev` split and evaluated **once** on the full `test` split —
a real held-out number, not cross-validated on the same data used for the earlier ablation.

**LLM feature engineering upgrade: Haiku → Sonnet, dev-only → dev+test.** Re-ran
`llm_feature_extraction.py --model claude-sonnet-5` on **both** dev and test (the original
experiment only ever covered dev — the "0.666" headline number from that run was never checked
against test.csv). Sonnet also turned out more reliable at following the JSON schema: it did not
reproduce Haiku's `dehumatizing`/`dehumanizing` key-typo (which affected 1074/1084 of Haiku's dev
responses).

**Gotcha worth flagging for anyone reusing this pipeline**: `llm_feature_extraction.py`'s
`build_sample()` calls `.reset_index(drop=True)` on the stratified sample before writing
`row_index` to the output JSONL — so `row_index` is the position **within the sample**, not the
original `dev.csv`/`test.csv` row. `build_ensemble.py`'s `sample_original_indices()` recomputes
the exact same sample (same seed, same selection logic, minus the final reset) to recover the
mapping back to the original row order the `probs_*.csv` files use. If `build_sample()` in
`llm_feature_extraction.py` ever changes, `sample_original_indices()` must change identically or
this silently misaligns rows (it fails loudly instead, via an assertion on the true label matching,
which is what caught this the first time).

**Results, all on the real test set** (re-verified against the corrected kernel v10 — see the
postscript above; these superseded a first pass whose PhoBERT-focal probabilities were corrupted
by the FocalLoss regression):

| Approach | Test macro-F1 |
|---|---|
| XLM-R alone (best single model) | 0.616 |
| 4-model ensemble, full test (6680 rows) | 0.627 |
| 4-model ensemble, OFFENSIVE/HATE-heavy subset (2546 rows, no LLM) | 0.706 |
| **4-model ensemble + Claude Sonnet features, same subset** | **0.735** |

Two things worth noting about interpreting this table: the subset rows use a different, harder
row population (all OFFENSIVE+HATE test rows + a 1.25x matched CLEAN sample — not the natural
83/7/10 distribution), which is why they score higher than the full-test ensemble row — they're
not directly comparable to it, only to each other. The fair "did the LLM features help" comparison
is 0.706 → 0.735 (+0.028), on identical rows, matching the methodology of the original Haiku
ablation. Full per-class breakdown is printed by `build_ensemble.py` and written to
`data/ensemble_ablation_results.json`.

## Open questions for further iteration (documented, not required for the CV-ready deliverable)

- Does a slang/teencode normalization pass before segmentation (e.g. `mk`→`mày`, elongated letters `đẹppp`→`đẹp`) reduce OOV subwords and help minority classes?
- Does thread-level context (parent post, not just the comment) reduce the OFFENSIVE/HATE ambiguity that single-comment models can't resolve?
- Multi-seed variance: this project reports single-seed results per stage; a rigorous follow-up would run 3 seeds and report mean ± std, especially for the rare HATE class where F1 is seed-sensitive on ~700 test examples.
