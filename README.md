# Pairwise gradient boosting vs TF-IDF logistic regression on CaseHOLD

## Motivation
The paper "Recognizing cited facts and principles in legal judgements" (2017) studies how to detect which facts and legal principles a judgement passage cites. Its corpus is not public here, so this experiment uses a public stand-in: CaseHOLD, where a citing context must be matched to the holding it relies on. It tests recognising a cited principle, not cited facts.

## Question
Can a pairwise-similarity gradient-boosting model with within-question ranking features beat a standard TF-IDF logistic-regression classifier at picking the correct cited holding for a judgement passage?

## Dataset
CaseHOLD from LexGLUE (`coastalcph/lex_glue`, `case_hold`), downloaded as parquet over HTTP from the Hugging Face hub. Each question has a citing context and 5 candidate holdings, one correct. A fixed random subsample of the official train split and of the official test split is used (seed 42). This is **not** the paper's own dataset.

## Models
- **Baseline:** TF-IDF word n-gram logistic regression on each (context, candidate) pair. The highest-scoring candidate per question is chosen.
- **Improved:** LightGBM on pairwise features: word and character TF-IDF cosine similarity, token overlap, lengths, and each candidate's rank, z-score and gap to the best within its question.

Both models use the same data, vectorisers, evaluation function and seeds (42, 43, 44). Hyperparameters are tuned by cross-validation on the training set only. The test set is used once at the end. Metrics are top-1 accuracy (main), macro-F1 on pair labels, and a bootstrap 95% CI of the accuracy difference.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the public CaseHOLD dataset (1,000-question test subsample), a standard TF-IDF logistic-regression text classifier reached 32.1% top-1 accuracy, while LightGBM with within-question ranking features reached 56.8% on the same test set, a gain of 24.7 points.

| Metric | Value |
|---|---|
| baseline_score | 0.321 |
| improved_score | 0.5677 |
| improvement | 0.2467 |
| baseline_score_std | 0 |
| improved_score_std | 0.0017 |
| baseline_f1 | 0.5756 |
| improved_f1 | 0.7298 |
| baseline_f1_std | 0 |
| improved_f1_std | 0.001062 |
| diff_ci95_low | 0.205 |
| diff_ci95_high | 0.289 |
| baseline_cv_acc | 0.3025 |
| improved_cv_acc | 0.6125 |
| random_chance_acc | 0.2 |
| runtime_seconds | 36.3 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in for the paper's non-public corpus; tests recognising a cited principle, not cited facts. Official CaseHOLD train/test splits are randomly subsampled. TF-IDF vectorisers are fitted on training text only (unsupervised, also used inside CV). Macro-F1 is on pair labels (chosen candidate=1). Bootstrap CI resamples test questions using per-question correctness averaged over seeds. Similarity features are lexical, so absolute accuracy is expected to be far below transformer models.

## Reference

This project is motivated by the following paper:

> O Shulayeva, A Siddharthan, A Wyner (2017). *Recognizing cited facts and principles in legal judgements*. Artificial Intelligence and Law. https://link.springer.com/article/10.1007/s10506-017-9197-6

Paper author: Prof. Advaith Siddharthan, University of Aberdeen (advaith@abdn.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
