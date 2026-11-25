# Technical report template

**Title:** Multilingual Fake News Detection Using Cross-Lingual Transformer Embeddings
**Author:** *(your name)*  **Institution / course / guide:** *(fill in)*  **Date:** *(fill in)*

> How to use this file: every `[PLACEHOLDER]` must be replaced with something you have actually produced or read. Do not paste numbers from `outputs_demo/` (synthetic data, stand-in encoder). Do not add a citation you have not read. Delete these instructions before submitting.

---

## Abstract

[PLACEHOLDER: 150–200 words. Problem, approach, dataset, main measured finding, main limitation. Write this last, from your real results.]

## 1. Introduction

- Real-world problem: misinformation spreads quickly on digital platforms, and not only in English. (See your research-introduction deck, slides 2–4.)
- Why multilingual detection is hard: the same claim in several languages, translation changes wording, informal / code-mixed text, uneven labelled data across languages, poor transfer between languages. (Slide 3.)
- Motivation in your own words. (Slide 8.)
- Research questions (keep the "may / could / investigate" wording until you have evidence):
  1. How well can multilingual Transformer sentence embeddings place the same claim, written in different languages, close together?
  2. Can a classifier trained on a higher-resource language (e.g. English) transfer to other languages?
  3. How does an embedding-based classifier compare with traditional NLP baselines (TF-IDF)?
  4. How does the approach behave on code-mixed or translated text? *(only if your data contains it)*
  5. Can similar-claim retrieval, or a generative model, produce explanations people find useful? *(usefulness is not measured by this code)*

## 2. Related work

[PLACEHOLDER: Your literature review. Cover (a) multilingual fake-news / fact-check datasets, (b) multilingual sentence encoders, (c) cross-lingual transfer for classification, (d) existing multilingual misinformation detection methods, (e) the research gap you identified.
Add only references you have read. Record each as: authors, year, title, venue / arXiv id, one-sentence relevance.]

## 3. Data

- Dataset used: [PLACEHOLDER: name, source link, version / access date, licence as stated on the record page].
- Size after cleaning: [PLACEHOLDER from `outputs/model/metadata.json` → `loading`]: rows, class counts, language counts, rows dropped (duplicates, conflicting labels, unmapped labels).
- How labels were mapped to Real (0) / Potentially Fake (1): [PLACEHOLDER: raw values → classes; mention any `--label-map` you wrote and why].
- What the text is: [PLACEHOLDER: e.g. fact-checkers' article text versus original posts. This changes what a model can learn.]
- Class imbalance: [PLACEHOLDER: share of the "fake" class].
- Known limitations of the dataset: [PLACEHOLDER: domain, time period, language mix, how labels were produced].

## 4. Method

### 4.1 Pipeline
Text → cleaning (Unicode normalisation, URL replacement, whitespace) → language identification (dataset column, else `langdetect`) → frozen multilingual sentence encoder → classifier → Real / Potentially Fake → retrieval-based explanation.

### 4.2 Encoder
[PLACEHOLDER: exact encoder name from `metadata.json` → `encoder`; languages it was trained on (from its model card); that it is used frozen and was not fine-tuned.]

### 4.3 Models compared
Majority class; TF-IDF word + logistic regression; TF-IDF character n-grams + logistic regression; embeddings + logistic regression (primary); embeddings + MLP; embeddings + HistGradientBoosting.

### 4.4 Training and selection
Balanced class weights where supported; regularisation strength `C` and the decision threshold chosen on the validation split only; the test split is used once. [PLACEHOLDER: selected hyper-parameters from `metadata.json` → `selected_hyperparameters`.]

### 4.5 Explanations
For selected predictions the system lists the most similar labelled training claims (any language). These are retrieval-based, not generative, and are not fact-checking. A prompt for a generative model is produced but **never sent** by this code.

## 5. Experimental protocol

- Split: [PLACEHOLDER: strategy printed by the run (grouped / chronological / stratified) and exact train / validation / test counts]. Explain why this split limits leakage (translations or duplicates of one claim stay together; or time order is respected).
- Metrics: macro-F1 (primary), fake-class precision / recall / F1, balanced accuracy, accuracy, ROC AUC. Explain why accuracy alone is misleading under imbalance.
- Cross-lingual transfer test: train on [PLACEHOLDER: source language], test on every language; compare with training on all languages.
- Alignment analysis: [PLACEHOLDER: run with `--parallel-csv` or state that no parallel data was available, so research question 1 is not answered].
- Single split and single seed: [PLACEHOLDER: state this, or add repeated runs and report the spread].
- Leakage tests: scalers / vocabularies / hyper-parameters / thresholds fitted on training or validation rows only; flipping every test label changes no test prediction (automated test).

## 6. Results

> Fill from `outputs/` only.

| Model | Macro-F1 | Fake-class F1 | Fake precision | Fake recall | Balanced acc. |
|---|---|---|---|---|---|
| [PLACEHOLDER] | | | | | |

- Per-language results: [PLACEHOLDER from `per_language_metrics.csv`; flag languages with too few test rows].
- Cross-lingual transfer: [PLACEHOLDER from `transfer_metrics.csv` and `plots/cross_lingual_transfer.png`].
- Alignment: [PLACEHOLDER from `alignment.csv`, or "not available"].
- Error analysis: [PLACEHOLDER: at least one false positive and one false negative from `explanations.md`, and what they suggest].

## 7. Discussion

- Which model performed best on your split, and by how much? Is the difference large compared with what a different split or seed could change? [PLACEHOLDER]
- Did the embeddings beat the TF-IDF baselines? If not, why might that be (domain, text type, language mix, frozen encoder)? [PLACEHOLDER]
- What do the transfer results say about research question 2? [PLACEHOLDER]
- What do the explanations show, and what can they not show? [PLACEHOLDER]

## 8. Limitations

- Labels come from the dataset's own annotation process; the model learns that labelling, not ground truth.
- Results are specific to this dataset, time period, split, seed and encoder.
- The encoder is frozen; fine-tuning might change the picture.
- [PLACEHOLDER: dataset-specific points, e.g. fact-check article text rather than original posts; domain; language coverage.]
- Code-mixed text and low-resource languages are only covered if your data contains them. [PLACEHOLDER]
- The explanations are retrieval-based; their usefulness to people was not measured.
- Nothing here shows the system works on live social-media content or for Indian-language misinformation in general.

## 9. Ethical considerations

- A classifier can wrongly flag true content, or miss false content. Present output as "potentially fake", never as a verdict.
- Verification against reliable external evidence stays with humans.
- A generative model is not a fact-checking source.

## 10. Conclusion and future work

[PLACEHOLDER: what you measured, honestly stated. Future work, e.g. a Hindi / regional-language dataset, fine-tuning the encoder, repeated splits, evidence retrieval, a user study of explanations.]

## References

[PLACEHOLDER: only sources you have read. Include the dataset citation and the encoder's model card / paper.]
