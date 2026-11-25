# Final submission checklist

## Before running
- [ ] Use Python 3.10–3.12.
- [ ] Create the virtual environment and install `requirements.txt`.
- [ ] Run `python -m unittest discover -s tests -v` and confirm all tests pass.
- [ ] Run `python run_project.py --source demo` once to confirm the software works offline.

## Real-data experiment
- [ ] Run `python run_project.py --source real` (internet needed for the dataset and the encoder model).
- [ ] Read the printed summary: columns used, raw label values, class counts, language counts, split strategy.
- [ ] If labels were mapped wrongly or too many rows were dropped, create a `--label-map` JSON and re-run.
- [ ] Confirm `outputs/model/metadata.json` shows dataset `fakecovid` (not `synthetic_demo`) and the encoder name you intended.
- [ ] Confirm `outputs/metrics.csv` contains all six models.
- [ ] Record the exact train/validation/test row counts and the split strategy.
- [ ] Compare macro-F1 and fake-class F1, not only accuracy; mention the class imbalance.
- [ ] Look at `per_language_metrics.csv`: note languages with too few test rows (`nan`).
- [ ] Look at `transfer_metrics.csv` and `plots/cross_lingual_transfer.png`.
- [ ] If you have parallel data, run with `--parallel-csv` and report `alignment.csv`.
- [ ] Read `explanations.md` and investigate at least one false positive and one false negative.

## Report and presentation
- [ ] Add the real results table and plots to your results section.
- [ ] Update the conclusion to name the actual best-performing model, even if it is a baseline.
- [ ] Keep the dataset citation, and check the dataset's Zenodo record for its licence and attribution requirements (README Section 2).
- [ ] Fill in `reports/technical_report_template.md` after your literature review; add only citations you have actually read.
- [ ] Read `reports/viva_questions.md` and be able to answer each question from your own results.
- [ ] State the limitations from README Section 12 (labels from fact-checkers, article text not original posts, COVID-only, 2020-only, frozen encoder, single split and seed).
- [ ] Do not describe the retrieval-based explanations as generative AI or as fact-checking.
- [ ] Keep "may / could / investigate" wording for anything you have not measured.
- [ ] Do not copy the synthetic demo scores into the final report or presentation.
- [ ] Replace `REPLACE WITH YOUR NAME` in `CITATION.cff` and add your institution, course and guide details to the cover page.

## Demonstration
- [ ] Keep the extracted project folder available.
- [ ] Run the real-data command before the presentation so the plots and metrics are current.
- [ ] Be prepared to explain the grouped/chronological split, the label-flipping leakage test, the baselines, the threshold tuning and the limitations.
