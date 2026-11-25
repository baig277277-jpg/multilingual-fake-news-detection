# Results - Multilingual Fake News Detection

Generated (UTC): 2026-11-13T04:10:52.831472+00:00
Dataset: synthetic_demo | Encoder: hashing-char-ngrams-512
Split: grouped - {'train': 303, 'validation': 66, 'test': 63}
Test class counts: {'Real': 33, 'Potentially Fake': 30}

> **SYNTHETIC DEMONSTRATION DATA.** Do not report any number below as a research result.

## Held-out test comparison

| Model | Macro_F1 | Fake_F1 | Fake_precision | Fake_recall | Balanced_accuracy | Accuracy | ROC_AUC |
|---|---|---|---|---|---|---|---|
| TF-IDF word + LogReg | 0.952 | 0.952 | 0.909 | 1.000 | 0.955 | 0.952 | 0.993 |
| TF-IDF char + LogReg | 0.952 | 0.952 | 0.909 | 1.000 | 0.955 | 0.952 | 1.000 |
| Embeddings + MLP | 0.952 | 0.952 | 0.909 | 1.000 | 0.955 | 0.952 | 0.969 |
| Embeddings + LogReg | 0.952 | 0.952 | 0.909 | 1.000 | 0.955 | 0.952 | 0.994 |
| Embeddings + HistGradientBoosting | 0.952 | 0.952 | 0.909 | 1.000 | 0.955 | 0.952 | 0.989 |
| Majority class | 0.344 | 0.000 | 0.000 | 0.000 | 0.500 | 0.524 | 0.500 |

Highest macro-F1 on this test split: **TF-IDF word + LogReg** (0.952). The proposed approach (Embeddings + LogReg) scored 0.952.

## Per-language results (all-language training)

| Model | language | n_test | fake_share | Macro_F1 |
|---|---|---|---|---|
| TF-IDF char + LogReg | en | 21 | 0.476 | 0.952 |
| TF-IDF char + LogReg | hi | 21 | 0.476 | 0.952 |
| TF-IDF char + LogReg | hi-latn | 21 | 0.476 | 0.952 |
| Embeddings + LogReg | en | 21 | 0.476 | 0.952 |
| Embeddings + LogReg | hi | 21 | 0.476 | 0.952 |
| Embeddings + LogReg | hi-latn | 21 | 0.476 | 0.952 |

Macro-F1 is blank (nan) for languages with too few test rows or a single class.

## Cross-lingual transfer (research question 2)

| Model | train_language | test_language | n_test | Macro_F1_trained_on_source_only | Macro_F1_trained_on_all_languages |
|---|---|---|---|---|---|
| TF-IDF char + LogReg | en | en | 21 | 0.952 | 0.952 |
| TF-IDF char + LogReg | en | hi | 21 | 0.323 | 0.952 |
| TF-IDF char + LogReg | en | hi-latn | 21 | 0.697 | 0.952 |
| Embeddings + LogReg | en | en | 21 | 0.952 | 0.952 |
| Embeddings + LogReg | en | hi | 21 | 0.714 | 0.952 |
| Embeddings + LogReg | en | hi-latn | 21 | 0.323 | 0.952 |

## Embedding alignment (research question 1)

| pivot_language | other_language | n_pairs | mean_cosine_true_pairs | mean_cosine_mismatched_pairs | top1_retrieval_accuracy | top1_chance |
|---|---|---|---|---|---|---|
| en | hi | 144 | 0.320 | 0.310 | 0.014 | 0.007 |
| en | hi-latn | 144 | 0.525 | 0.467 | 0.229 | 0.007 |

## Interpretation notes

- All models share one split. Scalers, TF-IDF vocabularies, hyperparameters (C) and decision thresholds are fitted or chosen on the training/validation rows only; the test split is used once for the numbers above.
- Macro-F1 and fake-class F1 are reported because the classes can be heavily imbalanced; accuracy alone can hide a model that predicts one class.
- Encoder status: no (offline smoke-test backend; character n-grams are not shared across scripts).
- A lower score for the embedding models than for a simple baseline is a valid finding and should be reported with possible reasons, not hidden.
- The 'explanations' are retrieval-based. They show similar labelled training claims and are not fact-checking.
- Labels reflect the dataset's own annotation process; the models learn that labelling, which may differ from the truth of a claim.
