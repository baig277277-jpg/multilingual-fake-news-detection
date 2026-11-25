# Multilingual Fake News Detection Using Cross-Lingual Transformer Embeddings

**Domain:** Natural Language Processing  
**Project owner:** *(add your name)*  
**Task:** Classify a claim or news text as *Real* or *Potentially Fake*, across languages  
**Main approach:** Frozen multilingual Transformer sentence embeddings + a small classifier, compared against traditional NLP baselines

> **Important before submission:** The bundled `outputs_demo/` folder was generated from a clearly labelled **synthetic** dataset (`data/demo_synthetic_multilingual.csv`) with an **offline, non-cross-lingual stand-in encoder**. It only proves that the software runs. Those numbers are **not research results**. Run the real-data command in Section 3 and use only those outputs in your report.

## 0. What has and has not been tested

| Item | Status |
|---|---|
| 16 automated tests (labels, cleaning, leakage-safe splits, alignment, explanations, end-to-end) | Passed (Python 3.13, pandas 3.0, scikit-learn 1.9) |
| Full pipeline on the synthetic demo with the offline `hashing` encoder | Ran successfully |
| Real multilingual Transformer encoder (`sentence-transformers`) | **Not run** by the author of this code: no network in the build environment. The code path is straightforward but untested. |
| Download of the FakeCovid dataset and its real column names | **Not run.** The URL and MD5 were taken from the Zenodo record page; the CSV's exact columns were not visible, so columns are auto-detected and printed (see Section 3). On re-checking, only the article count, language count and collection dates could be confirmed (via a library mirror); the licence, file name and MD5 could not. |
| Warning titles on demo plots | Added: every figure from the synthetic demo (or from the offline stand-in encoder) carries a red "NOT RESEARCH RESULTS" title. Real runs with a real encoder have no such title. |
| Automatic language identification with `langdetect` | **Not run** (only used when the CSV has no language column). |

If something fails on your first real run, the error messages tell you which column or option to pass. Fix that, rather than editing the models.

## 1. Project summary

The project asks whether embeddings from a pretrained **multilingual Transformer** let a fake-news classifier work across languages better than traditional methods that rely on the words themselves. It is a reproducible end-to-end pipeline, not just a model definition:

- Download and checksum verification of a public multilingual fact-check dataset (or load your own CSV).
- Cleaning, Unicode normalisation, duplicate and conflicting-label removal.
- Language identification step (uses the dataset's language column; falls back to `langdetect`).
- A **leakage-aware split** (grouped by claim, else chronological, else stratified random).
- Six models on one fixed split: a majority-class baseline, two TF-IDF baselines (word and character n-grams) and three heads on frozen cross-lingual embeddings (logistic regression, MLP, gradient boosting).
- A **cross-lingual transfer test**: train on English only, test on every language.
- An **embedding alignment analysis**: do translations of one claim land close together?
- Per-prediction explanations built from similar labelled training claims, plus an optional prompt for a generative model.
- Metrics, plots, a results report, saved models and metadata.
- Automated tests, including a check that changing the test labels does not change test predictions.

## 2. Dataset and source

**Default: FakeCovid (multilingual fact-checked COVID-19 news).**

- Zenodo record: <https://zenodo.org/records/3965871> (file `FakeCovid_July2020.csv`, about 55 MB, MD5 `c8f2774f9315f6311b2ab78e41ea9bea`, DOI 10.5281/zenodo.3965871).
- Paper: <https://arxiv.org/abs/2006.11343>.
- Licence: **check the Zenodo record page yourself** and keep whatever attribution it asks for in any submitted copy. An earlier version of this README said CC BY 4.0, but that could not be re-confirmed (the catalogue mirror that could be opened shows no licence), so do not rely on it.
- A library-catalogue mirror of the record lists 7,623 fact-checked news articles in 40 languages from 105 countries, collected 4 January – 1 July 2020 (written there as 04/01/2020 – 01/07/2020). The paper may describe an earlier version of the dataset, so counts in the paper and in the file can differ.
- The file name, size and MD5 above are **not independently confirmed**. If the MD5 does not match, the download stops with an error and tells you to download the CSV manually (nothing is silently accepted).
- Labels come from the fact-checking websites; the authors merged the original class names into a smaller set.

**Dataset citation:** Shahi, G. K., and Nandini, D. (2020). *FakeCovid – A Multilingual Cross-domain Fact Check News Dataset for COVID-19.* arXiv:2006.11343. Data: Zenodo, doi:10.5281/zenodo.3965871.

**Things to know about this dataset (they affect how you interpret results):**

- **Strong class imbalance.** In the paper's version, roughly four in five articles are rated "false". "Real" examples are few and varied. This is why the project reports macro-F1, fake-class F1 and balanced accuracy, and tunes the decision threshold on validation data. Accuracy alone would be misleading.
- **The text is the fact-checkers' article, not the original viral post.** A model may partly learn the style of fact-checking articles rather than the style of misinformation. Say so in your limitations.
- **COVID-19 only**, 2020 only, and the language mix is dominated by English.
- Annotated categories exist only for three languages in the paper, and duplicate removal was URL-based, so some duplicates may remain.
- The dataset is **not** a set of translations, so the alignment analysis (research question 1) needs a parallel file (Section 3).

**For the Indian-language focus:** *FactDRIL* (Singhal, Shah and Kumaraguru, IIIT-Delhi; <https://arxiv.org/abs/2102.11276>) covers English, Hindi and several regional Indian languages from IFCN-certified fact-checking websites. The paper says access is available on request, so it is **not** downloaded automatically. If you obtain it, use `--data` with your CSV (Section 3).

## 3. Run on Windows

Install Python 3.10–3.12. Extract the project, then double-click `setup_windows.bat`, or run in PowerShell from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the tests first:

```powershell
python -m unittest discover -s tests -v
```

Run the offline demonstration (synthetic data, stand-in encoder, not research results):

```powershell
python run_project.py --source demo
```

Run the real-data experiment. The first run downloads the dataset (about 55 MB) and the encoder model from Hugging Face (several hundred MB), so it needs internet and some time:

```powershell
python run_project.py --source real
```

**Check the printed summary before trusting any number.** The script prints the columns it chose, the raw label values it found, the class counts after cleaning and the language counts. If the label mapping looks wrong (for example, too many rows dropped, or "real" and "fake" swapped), write a small JSON file that maps the *exact* raw values printed to `1` (fake), `0` (real) or `null` (drop), then re-run:

```powershell
python run_project.py --source real --label-map my_labels.json
```

where `my_labels.json` looks like `{"False": 1, "True": 0}` (use the values from your own printout). Other useful options:

| Option | Purpose |
|---|---|
| `--text-col`, `--label-col`, `--language-col`, `--date-col`, `--group-col` | Choose columns explicitly if auto-detection picked the wrong one |
| `--data path\to\file.csv` | Use your own dataset (text, label, language columns) |
| `--parallel-csv path\to\pairs.csv` | CSV with columns `text`, `language`, `claim_id` (translations share a `claim_id`) for the alignment analysis |
| `--encoder <name>` | Try another encoder, e.g. `sentence-transformers/LaBSE` (check the model card for language coverage) |
| `--split grouped\|chronological\|stratified` | Override the automatic split choice |
| `--date-dayfirst` | Read dates like 04/01/2020 as day/month |
| `--transfer-source hi` | Train the transfer test on another language |

If the automatic download fails, download `FakeCovid_July2020.csv` from the Zenodo link and save it as `data/real/FakeCovid_July2020.csv`, then re-run.

## 4. Tests

```powershell
python -m unittest discover -s tests -v
```

The tests verify label normalisation (including negations such as "not true" and dropping ambiguous verdicts), URL and Unicode cleaning, that Devanagari words are not split by the tokenizer, duplicate and conflicting-label handling, that grouped splits never put one claim in two splits, that chronological splits respect time, that alignment scores are exactly 1 and 0 for a perfectly aligned toy encoder, that explanations only retrieve training claims, and that **flipping every test label does not change a single test prediction** (so test labels cannot be leaking into training, tuning or thresholds).

## 5. Protocol and leakage control

- **One split for every model**, 70% training / 15% validation / 15% test.
- **Split choice (`auto`)**: rows that share a `claim_id` (translations or duplicates of one claim) are kept in the same split. Without `claim_id`, a chronological split is used when dates parse for at least 90% of rows. Otherwise a stratified random split is used and the run prints a warning, because translated or near-duplicate claims can then straddle the split and inflate scores.
- Scalers, TF-IDF vocabularies, the regularisation strength `C` and the decision threshold are fitted or chosen on training/validation rows only. The test split is used once for the final numbers.
- Exact duplicate texts are removed, and texts that appear with conflicting labels are dropped.
- Class imbalance is handled with balanced class weights where the model supports them and with a validation-tuned threshold for every probabilistic model. The majority baseline shows what "always predict the common class" scores.

## 6. Models compared

| Model | Purpose |
|---|---|
| Majority class | Floor: always predicts the most common training class |
| TF-IDF word + LogReg | Traditional bag-of-words style baseline (custom tokenizer that works for Devanagari) |
| TF-IDF char + LogReg | Strong traditional multilingual baseline using character n-grams |
| Embeddings + LogReg | **Proposed approach** (reference model for plots, confusion matrix and explanations) |
| Embeddings + MLP | Non-linear head on the same embeddings |
| Embeddings + HistGradientBoosting | Tree-based head on the same embeddings |

The encoder is a **pretrained multilingual sentence encoder used as-is (frozen, not fine-tuned)**. The default is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Check its model card to confirm it covers the languages in your data. Fine-tuning a Transformer end to end is a natural extension but is not part of this project. The embedding approach is **not assumed to win**. If TF-IDF with character n-grams beats it on your data, report that and discuss why.

## 7. How the outputs map to your research questions

| Research question | Where to look | Caveat |
|---|---|---|
| 1. How well do embeddings represent the same claim across languages? | `alignment.csv`, `plots/embedding_alignment.png` | Needs translations: automatic only when the data has a `claim_id` shared across languages, otherwise pass `--parallel-csv`. Skipped for FakeCovid alone. |
| 2. Does knowledge transfer from high-resource to lower-resource languages? | `transfer_metrics.csv`, `plots/cross_lingual_transfer.png` | Trains on one language (English by default) and tests on all. Languages with fewer than 20 test rows show `nan`. |
| 3. How does it compare with traditional NLP/ML? | `metrics.csv`, `plots/model_comparison.png` | Single split and seed; differences between close models may be noise. |
| 4. Code-mixed or translated misinformation? | `per_language_metrics.csv` | Only measured if your data contains such text labelled with its own language code. FakeCovid is not specifically code-mixed, and the demo's romanised-Hindi rows are template text, not real code-mixing. |
| 5. Can Generative AI give useful explanations? | `explanations.md`, `explanations.csv` | See Section 8. Usefulness to people is **not measured** here; that needs manual rating or a user study. |

## 8. What the "Generative AI" part is, and is not

The explanations written by this code are **retrieval-based**: for the most confident predictions in each outcome cell (true/false positive/negative), they list the most similar *labelled training claims* in any language. That is not generative AI, and it is not fact-checking. It only shows what the embedding space considers similar.

`explanations.csv` also contains an `llm_prompt` column. It is a ready-made prompt for a generative model that instructs it to use only the given information, not to declare the claim true or false, and to say that external verification is needed. **This project never calls a generative model.** If you try one, treat its output as a draft explanation, check it by hand, and report honestly how you evaluated it. A generative model is not a fact-checking source.

## 9. Output files

After a successful run, `outputs/` (or `outputs_demo/`) contains:

- `metrics.csv` – model comparison (macro-F1, fake-class precision/recall/F1, balanced accuracy, accuracy, ROC AUC).
- `per_language_metrics.csv` – the same models broken down by language.
- `transfer_metrics.csv` – train-on-one-language versus train-on-all comparison.
- `alignment.csv` – translation-pair similarity and retrieval accuracy (when parallel data exists).
- `test_predictions.csv` – scores and predictions of every model on the test split.
- `explanations.md`, `explanations.csv` – example outputs with neighbours and prompts.
- `report_results.md` – auto-generated results table with interpretation notes.
- `plots/` – `model_comparison.png`, `per_language_f1.png`, `cross_lingual_transfer.png`, `embedding_alignment.png`, `confusion_matrix.png`.
- `model/` – one `.joblib` file per model and `metadata.json` (data source, columns used, split counts, hyperparameters, package versions).

## 10. Suggested demonstration

1. Open the dataset record and explain where labels come from.
2. Show the printed column and label summary, then the class imbalance.
3. Run the automated tests, and explain the label-flipping leakage test.
4. Run `python run_project.py --source real`.
5. Open `metrics.csv` and explain macro-F1 versus accuracy.
6. Show `per_language_metrics.csv` and `cross_lingual_transfer.png`.
7. Explain which model won and why the embedding approach did or did not beat TF-IDF.
8. Open `explanations.md` and discuss a false positive and a false negative.
9. Discuss limitations (Section 12).

## 11. Other folders (report, viva, presentation)

| Path | What it is |
|---|---|
| `reports/technical_report_template.md` | Section-by-section report skeleton with explicit placeholders for results, citations and limitations. Nothing in it is a result. |
| `reports/viva_questions.md` | Likely examiner questions with short answer outlines tied to this code. |
| `presentation/Multilingual Fake News Detection - Research Introduction.pptx` | The 15-slide research-introduction deck (problem, motivation, preliminary research questions). It contains **no results**; add a results slide only after a real run. |

## 12. Responsible reporting

- Do not report synthetic demo numbers as results.
- Labels are the dataset's own, produced by fact-checkers; the model learns their labelling, not ground truth.
- Results are specific to this dataset (COVID-19, 2020, fact-check articles), split and encoder. They do not show that the system works on live social-media posts, on code-mixed text, or for Indian-language misinformation in general.
- A system like this can wrongly flag true claims. Present its output as "potentially fake", never as a verdict, and keep human verification with reliable external evidence in the loop.
- Report the measured ranking even if a simple baseline wins.
