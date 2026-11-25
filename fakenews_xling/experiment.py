"""Controlled model comparison, cross-lingual transfer test, metrics, plots and report."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)

from .alignment import alignment_report, choose_pivot
from .data import (
    FAKECOVID_PAPER, FAKECOVID_RECORD, LABEL_NAMES, load_dataset_csv, load_parallel_csv, split_frame,
)
from .embeddings import DEFAULT_ENCODER, encode_with_cache, get_encoder
from .explain import build_explanations, explanations_to_markdown
from .models import train_embedding_model, train_text_model

MAJORITY = "Majority class"
TFIDF_WORD = "TF-IDF word + LogReg"
TFIDF_CHAR = "TF-IDF char + LogReg"
EMB_LOGREG = "Embeddings + LogReg"
EMB_MLP = "Embeddings + MLP"
EMB_HGB = "Embeddings + HistGradientBoosting"
PRIMARY = EMB_LOGREG  # proposed approach; it is NOT assumed to win


@dataclass
class ExperimentConfig:
    seed: int = 42
    encoder: str = DEFAULT_ENCODER
    batch_size: int = 64
    split: str = "auto"
    transfer_source: str | None = None
    min_language_rows: int = 20
    dayfirst: bool = False


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def metric_row(name: str, y_true, y_pred, proba) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Fake_precision": float(precision[0]),
        "Fake_recall": float(recall[0]),
        "Fake_F1": float(f_score[0]),
        "ROC_AUC": auc,
    }


def _group_macro_f1(y_true, y_pred, min_rows: int) -> float:
    if len(y_true) < min_rows or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _transfer_experiment(source, train, val, test, E_tr, E_v, E_te, preds, config):
    """Train on one language only, test on every language (zero-shot for the others)."""
    mask_tr = (train["language"] == source).to_numpy()
    mask_v = (val["language"] == source).to_numpy()
    ytr, yv = train["label"].to_numpy()[mask_tr], val["label"].to_numpy()[mask_v]
    if mask_tr.sum() < 50 or len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2:
        return pd.DataFrame(), f"Skipped: too little data or a single class in '{source}' training/validation rows."

    candidates = {}
    pipe, meta = train_text_model(
        "char", train["text"].to_numpy()[mask_tr].tolist(), ytr, val["text"].to_numpy()[mask_v].tolist(), yv, config.seed
    )
    candidates[TFIDF_CHAR] = (lambda idx, p=pipe: p.predict_proba(test["text"].to_numpy()[idx].tolist())[:, 1], meta["threshold"])
    pipe, meta = train_embedding_model("logreg", E_tr[mask_tr], ytr, E_v[mask_v], yv, config.seed)
    candidates[EMB_LOGREG] = (lambda idx, p=pipe: p.predict_proba(E_te[idx])[:, 1], meta["threshold"])

    y_test = test["label"].to_numpy()
    rows = []
    for name, (predict, threshold) in candidates.items():
        for language in sorted(test["language"].unique()):
            idx = np.where((test["language"] == language).to_numpy())[0]
            f_source = _group_macro_f1(y_test[idx], (predict(idx) >= threshold).astype(int), config.min_language_rows)
            f_all = _group_macro_f1(y_test[idx], preds[name][idx], config.min_language_rows)
            rows.append(
                {
                    "Model": name,
                    "train_language": source,
                    "test_language": language,
                    "n_test": int(len(idx)),
                    "Macro_F1_trained_on_source_only": f_source,
                    "Macro_F1_trained_on_all_languages": f_all,
                    "delta_source_only_minus_all": f_source - f_all,
                }
            )
    return pd.DataFrame(rows), ""


def _mark(fig, watermark: str) -> None:
    """Put a red warning title on a figure so demo plots cannot be mistaken for results."""
    if watermark:
        fig.suptitle(watermark, color="firebrick", fontsize=9, fontweight="bold")


def _save_figures(metrics, per_language, transfer, alignment, y_test, preds, outdir: Path, watermark: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("Macro_F1")
    fig, ax = plt.subplots(figsize=(9, max(3.8, 0.5 * len(ordered))))
    ax.barh(ordered["Model"], ordered["Macro_F1"])
    ax.set_xlabel("Macro-F1 on the held-out test split (higher is better)")
    ax.set_title("Model comparison")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    _mark(fig, watermark)
    fig.tight_layout()
    fig.savefig(outdir / "model_comparison.png", dpi=180)
    plt.close(fig)

    shown = per_language[per_language["Model"].isin([PRIMARY, TFIDF_CHAR, TFIDF_WORD])].dropna(subset=["Macro_F1"])
    if not shown.empty:
        pivot = shown.pivot(index="language", columns="Model", values="Macro_F1")
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(pivot) + 3), 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Macro-F1 (languages with enough test rows only)")
        ax.set_title("Per-language performance (trained on all languages)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=0)
        _mark(fig, watermark)
        fig.tight_layout()
        fig.savefig(outdir / "per_language_f1.png", dpi=180)
        plt.close(fig)

    if transfer is not None and not transfer.empty:
        usable = transfer.dropna(subset=["Macro_F1_trained_on_source_only"])
        if not usable.empty:
            pivot = usable.pivot(index="test_language", columns="Model", values="Macro_F1_trained_on_source_only")
            fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(pivot) + 3), 4.5))
            pivot.plot(kind="bar", ax=ax)
            source = str(usable["train_language"].iloc[0])
            ax.set_ylabel("Macro-F1")
            ax.set_title(f"Trained on '{source}' only, tested on each language")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=0)
            _mark(fig, watermark)
            fig.tight_layout()
            fig.savefig(outdir / "cross_lingual_transfer.png", dpi=180)
            plt.close(fig)

    if alignment is not None and not alignment.empty:
        positions = np.arange(len(alignment))
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(alignment) + 3), 4.5))
        ax.bar(positions - 0.2, alignment["mean_cosine_true_pairs"], width=0.4, label="True translation pairs")
        ax.bar(positions + 0.2, alignment["mean_cosine_mismatched_pairs"], width=0.4, label="Mismatched pairs")
        ax.set_xticks(positions)
        ax.set_xticklabels(alignment["pivot_language"] + " ↔ " + alignment["other_language"])
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Embedding alignment between languages")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        _mark(fig, watermark)
        fig.tight_layout()
        fig.savefig(outdir / "embedding_alignment.png", dpi=180)
        plt.close(fig)

    matrix = confusion_matrix(y_test, preds[PRIMARY], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL_NAMES[0], LABEL_NAMES[1]])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([LABEL_NAMES[0], LABEL_NAMES[1]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{PRIMARY}: confusion matrix")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    _mark(fig, watermark)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def run_experiment(
    data_path, output_dir, config: ExperimentConfig | None = None, dataset_name: str = "custom",
    columns: dict | None = None, label_map: dict | None = None, parallel_path=None, cache_dir=None,
):
    config = config or ExperimentConfig()
    output_dir = Path(output_dir)
    model_dir, plot_dir = output_dir / "model", output_dir / "plots"
    cache_dir = Path(cache_dir) if cache_dir else output_dir / "cache"
    model_dir.mkdir(parents=True, exist_ok=True)

    frame, load_info = load_dataset_csv(data_path, label_map=label_map, dayfirst=config.dayfirst, **(columns or {}))
    train, val, test, split_info = split_frame(frame, config.split, config.seed)

    backend = get_encoder(config.encoder)
    E_all = encode_with_cache(backend, frame["text"].tolist(), cache_dir, config.batch_size)
    E_tr, E_v, E_te = (E_all[part.index.to_numpy()] for part in (train, val, test))
    y_tr, y_v, y_te = (part["label"].to_numpy() for part in (train, val, test))

    fitted, probas, thresholds, train_meta = {}, {}, {}, {}

    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(y_tr), 1)), y_tr)
    probas[MAJORITY] = dummy.predict_proba(np.zeros((len(y_te), 1)))[:, 1]
    thresholds[MAJORITY] = 0.5
    fitted[MAJORITY] = dummy

    for name, kind in ((TFIDF_WORD, "word"), (TFIDF_CHAR, "char")):
        pipe, meta = train_text_model(kind, train["text"].tolist(), y_tr, val["text"].tolist(), y_v, config.seed)
        fitted[name], train_meta[name], thresholds[name] = pipe, meta, meta["threshold"]
        probas[name] = pipe.predict_proba(test["text"].tolist())[:, 1]

    for name, kind in ((EMB_LOGREG, "logreg"), (EMB_MLP, "mlp"), (EMB_HGB, "hgb")):
        pipe, meta = train_embedding_model(kind, E_tr, y_tr, E_v, y_v, config.seed)
        fitted[name], train_meta[name], thresholds[name] = pipe, meta, meta["threshold"]
        probas[name] = pipe.predict_proba(E_te)[:, 1]

    preds = {name: (probas[name] >= thresholds[name]).astype(int) for name in probas}

    metrics = pd.DataFrame([metric_row(n, y_te, preds[n], probas[n]) for n in probas]).sort_values(
        "Macro_F1", ascending=False
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    language_rows = []
    for name in probas:
        for language in sorted(test["language"].unique()):
            idx = np.where((test["language"] == language).to_numpy())[0]
            language_rows.append(
                {
                    "Model": name,
                    "language": language,
                    "n_test": int(len(idx)),
                    "fake_share": float(y_te[idx].mean()) if len(idx) else float("nan"),
                    "Accuracy": float(accuracy_score(y_te[idx], preds[name][idx])) if len(idx) else float("nan"),
                    "Macro_F1": _group_macro_f1(y_te[idx], preds[name][idx], config.min_language_rows),
                }
            )
    per_language = pd.DataFrame(language_rows)
    per_language.to_csv(output_dir / "per_language_metrics.csv", index=False)

    prediction_frame = pd.DataFrame(
        {"text": test["text"].to_numpy(), "language": test["language"].to_numpy(), "true_label": y_te}
    )
    if "claim_id" in test:
        prediction_frame.insert(0, "claim_id", test["claim_id"].to_numpy())
    for name in probas:
        prediction_frame[f"{_slug(name)}_score"] = probas[name]
        prediction_frame[f"{_slug(name)}_pred"] = preds[name]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    source = config.transfer_source or ("en" if (train["language"] == "en").any() else str(train["language"].value_counts().idxmax()))
    transfer, transfer_note = _transfer_experiment(source, train, val, test, E_tr, E_v, E_te, preds, config)
    if not transfer.empty:
        transfer.to_csv(output_dir / "transfer_metrics.csv", index=False)

    alignment, alignment_note = pd.DataFrame(), ""
    parallel_frame = None
    if parallel_path:
        parallel_frame = load_parallel_csv(parallel_path)
    elif "claim_id" in frame and frame.groupby("claim_id")["language"].nunique().max() > 1:
        parallel_frame = frame
    if parallel_frame is not None:
        pivot = choose_pivot(parallel_frame, config.transfer_source)
        alignment = alignment_report(backend, parallel_frame, pivot, cache_dir, config.batch_size, seed=config.seed)
        if alignment.empty:
            alignment_note = "No language pairs with at least 5 parallel claims were found."
        else:
            alignment.to_csv(output_dir / "alignment.csv", index=False)
    else:
        alignment_note = "Skipped: the data has no claim_id shared across languages and no --parallel-csv was given."

    items = build_explanations(test, train, E_te, E_tr, probas[PRIMARY], thresholds[PRIMARY])
    (output_dir / "explanations.md").write_text(explanations_to_markdown(items), encoding="utf-8")
    pd.DataFrame(
        [{**{k: v for k, v in item.items() if k != "neighbours"},
          "neighbours": " | ".join(f"[{n['language']}, {n['label']}, {n['similarity']:.2f}] {n['text']}" for n in item["neighbours"])}
         for item in items]
    ).to_csv(output_dir / "explanations.csv", index=False, encoding="utf-8")

    for name, estimator in fitted.items():
        joblib.dump(estimator, model_dir / f"{_slug(name)}.joblib")

    is_demo = dataset_name == "synthetic_demo"
    meta = {
        "project": "Multilingual Fake News Detection Using Cross-Lingual Transformer Embeddings",
        "dataset": dataset_name,
        "dataset_type": "synthetic_demo" if is_demo else "user_or_public_dataset",
        "data_path": Path(data_path).as_posix(),
        "encoder": backend.name,
        "encoder_cross_lingual_status": backend.cross_lingual_status,
        "loading": load_info,
        "split": split_info,
        "config": asdict(config),
        "selected_hyperparameters": train_meta,
        "primary_model": PRIMARY,
        "transfer_source_language": source,
        "transfer_note": transfer_note,
        "alignment_note": alignment_note,
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": _package_version("scikit-learn"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "SYNTHETIC DEMONSTRATION DATA: results only show that the software runs."
            if is_demo else
            "Results are specific to this dataset, split and encoder. Labels come from the dataset "
            "and are not independently verified; do not present them as universal performance."
        ),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    if is_demo:
        watermark = "SYNTHETIC DEMO DATA - NOT RESEARCH RESULTS"
    elif str(backend.cross_lingual_status).lower().startswith("no"):
        watermark = "OFFLINE STAND-IN ENCODER (NOT CROSS-LINGUAL) - NOT RESEARCH RESULTS"
    else:
        watermark = ""
    _save_figures(metrics, per_language, transfer, alignment, y_te, preds, plot_dir, watermark=watermark)
    _write_results_report(output_dir, metrics, per_language, transfer, alignment, meta)
    return metrics, meta


def _markdown_table(df: pd.DataFrame, columns: list[str], float_format: str = "{:.3f}") -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(float_format.format(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_results_report(output_dir, metrics, per_language, transfer, alignment, meta):
    best = metrics.iloc[0]
    primary = metrics[metrics["Model"] == PRIMARY].iloc[0]
    lines = [
        "# Results - Multilingual Fake News Detection",
        "",
        f"Generated (UTC): {meta['generated_at_utc']}",
        f"Dataset: {meta['dataset']} | Encoder: {meta['encoder']}",
        f"Split: {meta['split']['strategy']} - {meta['split']['counts']}",
        f"Test class counts: {meta['split']['class_counts']['test']}",
        "",
    ]
    if meta["dataset_type"] == "synthetic_demo":
        lines += ["> **SYNTHETIC DEMONSTRATION DATA.** Do not report any number below as a research result.", ""]
    lines += ["## Held-out test comparison", ""]
    lines += _markdown_table(metrics, ["Model", "Macro_F1", "Fake_F1", "Fake_precision", "Fake_recall", "Balanced_accuracy", "Accuracy", "ROC_AUC"])
    lines += [
        "",
        f"Highest macro-F1 on this test split: **{best['Model']}** ({best['Macro_F1']:.3f}). "
        f"The proposed approach ({PRIMARY}) scored {primary['Macro_F1']:.3f}.",
        "",
        "## Per-language results (all-language training)",
        "",
    ]
    shown = per_language[per_language["Model"].isin([PRIMARY, TFIDF_CHAR])]
    lines += _markdown_table(shown, ["Model", "language", "n_test", "fake_share", "Macro_F1"])
    lines += ["", "Macro-F1 is blank (nan) for languages with too few test rows or a single class.", ""]

    lines += ["## Cross-lingual transfer (research question 2)", ""]
    if transfer is not None and not transfer.empty:
        lines += _markdown_table(transfer, ["Model", "train_language", "test_language", "n_test", "Macro_F1_trained_on_source_only", "Macro_F1_trained_on_all_languages"])
    else:
        lines += [meta["transfer_note"] or "Not run."]
    lines += ["", "## Embedding alignment (research question 1)", ""]
    if alignment is not None and not alignment.empty:
        lines += _markdown_table(alignment, ["pivot_language", "other_language", "n_pairs", "mean_cosine_true_pairs", "mean_cosine_mismatched_pairs", "top1_retrieval_accuracy", "top1_chance"])
    else:
        lines += [meta["alignment_note"] or "Not run."]

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- All models share one split. Scalers, TF-IDF vocabularies, hyperparameters (C) and decision "
        "thresholds are fitted or chosen on the training/validation rows only; the test split is used once "
        "for the numbers above.",
        "- Macro-F1 and fake-class F1 are reported because the classes can be heavily imbalanced; accuracy alone can hide a "
        "model that predicts one class.",
        "- Encoder status: " + meta["encoder_cross_lingual_status"] + ".",
        "- A lower score for the embedding models than for a simple baseline is a valid finding and should be "
        "reported with possible reasons, not hidden.",
        "- The 'explanations' are retrieval-based. They show similar labelled training claims and are not fact-checking.",
        "- Labels reflect the dataset's own annotation process; the models learn that labelling, which may differ "
        "from the truth of a claim.",
        "",
    ]
    (output_dir / "report_results.md").write_text("\n".join(lines), encoding="utf-8")
