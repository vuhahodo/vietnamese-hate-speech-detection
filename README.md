---
permalink: nlp-commerce-doctor/readme
title: README
type: note
---

# Vietnamese Spam Review Detection with Multi-objective Evolutionary Optimization

Research project on [ViSpamReviews](https://github.com/sonlam1102/vispamdetection): classical feature blocks and classifiers, with NSGA-II / multi-objective DE selecting feature blocks, hyperparameters and ensemble weights. The Pareto front trades macro F1 against feature count (inference cost).

The plan, tasks and completion criteria are in [docs/PLAN.md](docs/PLAN.md).

## Data

ViSpamReviews is CC-BY-NC-4.0 with gated access. It is downloaded locally into `data/` and is never committed. Only code and aggregated results are in this repository.

## Evaluation rule

Model and search decisions use cross-validation on train and the dev split. The test split is scored once, for the final configurations.

## Previous project

The earlier ViHSD hate speech project is preserved at the git tag `archive/vihsd-hate-speech`.