---
permalink: nlp-commerce-doctor/readme
title: README
type: note
---

# Vietnamese Spam Review Detection with Multi-objective Evolutionary Optimization

Research project on [ViSpamReviews](https://github.com/sonlam1102/vispamdetection): classical feature blocks and classifiers, with NSGA-II / multi-objective DE selecting feature blocks, hyperparameters and ensemble weights. The Pareto front trades macro F1 against feature count (inference cost).

The plan, tasks and completion criteria are in [docs/PLAN.md](docs/PLAN.md).

## Main result (task T1, test split, 3,971 reviews)

| Model | Features | Macro F1 [95% CI] |
|---|---:|---:|
| PhoBERT-base-v2 | n/a | 0.8945 [0.884, 0.905] |
| Classical, EA-selected (multi-objective DE knee) | 24,060 | 0.8630 [0.851, 0.874] |
| Classical, all four blocks | 149,456 | 0.8619 [0.850, 0.874] |
| Char TF-IDF + LR (VeriReview baseline) | 113,258 | 0.8355 [0.822, 0.849] |

The EA-selected model keeps the accuracy of the full feature set with 16% of the features;
multi-objective search beats single-objective GA and random/grid on hypervolume but not on
accuracy. Full report: [docs/report/REPORT.md](docs/report/REPORT.md).

## Data

ViSpamReviews is CC-BY-NC-4.0 with gated access. It is downloaded locally into `data/` and is never committed. Only code and aggregated results are in this repository.

## Evaluation rule

Model and search decisions use cross-validation on train and the dev split. The test split is scored once, for the final configurations.

## Previous project

The earlier ViHSD hate speech project is preserved at the git tag `archive/vihsd-hate-speech`.