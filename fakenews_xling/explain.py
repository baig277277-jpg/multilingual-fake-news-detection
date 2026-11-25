"""Human-readable output for each prediction (research question 5).

Two layers:
1. A retrieval-based explanation that is built here without any generative model: it lists
   the most similar *labelled training claims* (in any language). It shows what the
   embedding space considers similar. It is NOT evidence about whether a claim is true.
2. An optional prompt for a generative model (build_llm_prompt). This module never calls a
   model itself; the prompt is a starting point for the 'Generative AI explanation layer'
   in the proposal and the model's answer would still need checking against real evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABEL_NAMES


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _snippet(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def nearest_training_examples(query_emb: np.ndarray, train_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = _unit(train_emb) @ _unit(query_emb.reshape(1, -1)).ravel()
    top = np.argsort(-sims)[:k]
    return top, sims[top]


def build_llm_prompt(text: str, language: str, label_name: str, score: float, neighbours: list[dict]) -> str:
    lines = [
        "You are helping a student explain the output of a fake-news classifier.",
        "Use ONLY the information below. Do not say whether the claim is true or false, and do not",
        "invent facts, sources or statistics. If the information is insufficient, say so and state that",
        "the claim must be checked against reliable external sources (fact-checkers, official records).",
        "",
        f"Claim ({language}): {text}",
        f"Classifier output: {label_name} (score {score:.2f})",
        "Most similar labelled training claims:",
    ]
    for item in neighbours:
        lines.append(f"- [{item['language']}, labelled {item['label']}, similarity {item['similarity']:.2f}] {item['text']}")
    lines.append("")
    lines.append("Write a short explanation (at most 4 sentences) a non-expert can follow.")
    return "\n".join(lines)


def build_explanations(
    test_frame: pd.DataFrame, train_frame: pd.DataFrame, E_test: np.ndarray, E_train: np.ndarray,
    proba: np.ndarray, threshold: float, k: int = 3, per_cell: int = 3,
) -> list[dict]:
    """Explain the most confident predictions in each outcome cell (TP, TN, FP, FN)."""
    y_true = test_frame["label"].to_numpy()
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    confidence = np.abs(np.asarray(proba) - threshold)
    cells = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
    train_text = train_frame["text"].to_numpy()
    train_lang = train_frame["language"].to_numpy()
    train_label = train_frame["label"].to_numpy()
    results = []
    for outcome, (true_value, pred_value) in cells.items():
        candidates = np.where((y_true == true_value) & (y_pred == pred_value))[0]
        for position in candidates[np.argsort(-confidence[candidates])][:per_cell]:
            top, sims = nearest_training_examples(E_test[position], E_train, k)
            neighbours = [
                {
                    "language": str(train_lang[i]),
                    "label": LABEL_NAMES[int(train_label[i])],
                    "similarity": float(s),
                    "text": _snippet(str(train_text[i])),
                }
                for i, s in zip(top, sims)
            ]
            n_fake = sum(item["label"] == LABEL_NAMES[1] for item in neighbours)
            pred_name = LABEL_NAMES[int(pred_value)]
            text = str(test_frame["text"].iloc[position])
            language = str(test_frame["language"].iloc[position])
            explanation = (
                f"Prediction: {pred_name} (score {proba[position]:.2f}; decision threshold {threshold:.2f}). "
                f"{n_fake} of the {k} most similar labelled training claims are '{LABEL_NAMES[1]}'. "
                "This only describes similarity inside the model's embedding space; it is not evidence about "
                "whether the claim is true. Verify with reliable external sources."
            )
            results.append(
                {
                    "outcome": outcome,
                    "language": language,
                    "text": _snippet(text, 300),
                    "true_label": LABEL_NAMES[int(true_value)],
                    "predicted_label": pred_name,
                    "score_potentially_fake": float(proba[position]),
                    "neighbours": neighbours,
                    "explanation": explanation,
                    "llm_prompt": build_llm_prompt(_snippet(text, 300), language, pred_name, float(proba[position]), neighbours),
                }
            )
    return results


def explanations_to_markdown(items: list[dict]) -> str:
    lines = [
        "# Example outputs (retrieval-based explanations)",
        "",
        "Selected as the most confident predictions in each outcome cell: TP = correctly flagged, "
        "TN = correctly passed, FP = real claim flagged, FN = fake claim missed. These come from the "
        "held-out test split. Neighbours are labelled *training* claims. No generative model was used.",
        "",
    ]
    for item in items:
        lines += [
            f"## {item['outcome']} - {item['language']}",
            "",
            f"> {item['text']}",
            "",
            f"- True label: **{item['true_label']}**; predicted: **{item['predicted_label']}** "
            f"(score {item['score_potentially_fake']:.2f})",
            f"- {item['explanation']}",
            "- Most similar labelled training claims:",
        ]
        for n in item["neighbours"]:
            lines.append(f"  - [{n['language']}, {n['label']}, similarity {n['similarity']:.2f}] {n['text']}")
        lines.append("")
    return "\n".join(lines)
