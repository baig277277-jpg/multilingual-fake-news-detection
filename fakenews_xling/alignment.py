"""Does the encoder place translations of the same claim close together? (research question 1)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .embeddings import encode_with_cache


def choose_pivot(frame: pd.DataFrame, preferred: str | None = None) -> str:
    languages = frame["language"].value_counts()
    if preferred and preferred in languages.index:
        return preferred
    return "en" if "en" in languages.index else str(languages.index[0])


def build_pairs(frame: pd.DataFrame, pivot: str) -> dict[str, tuple[list[str], list[str]]]:
    """Pair the pivot-language text of each claim_id with its text in every other language."""
    first = frame.drop_duplicates(["claim_id", "language"])
    pivot_text = first[first["language"] == pivot].set_index("claim_id")["text"]
    pairs = {}
    for language, part in first[first["language"] != pivot].groupby("language"):
        other = part.set_index("claim_id")["text"]
        common = pivot_text.index.intersection(other.index)
        if len(common) >= 5:
            pairs[str(language)] = (pivot_text.loc[common].tolist(), other.loc[common].tolist())
    return pairs


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def alignment_report(
    backend, frame: pd.DataFrame, pivot: str, cache_dir, batch_size: int = 64,
    max_pairs: int = 1500, seed: int = 42,
) -> pd.DataFrame:
    """Compare cosine similarity of true translation pairs with mismatched pairs.

    top-1 retrieval: for each pivot-language text, is its own translation the most
    similar candidate among all candidates in the other language? Chance is 1/n.
    This uses no labels, so using every row of the data is not a leakage risk.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for language, (texts_a, texts_b) in build_pairs(frame, pivot).items():
        if len(texts_a) > max_pairs:
            keep = np.sort(rng.choice(len(texts_a), size=max_pairs, replace=False))
            texts_a, texts_b = [texts_a[i] for i in keep], [texts_b[i] for i in keep]
        emb_a = _unit(encode_with_cache(backend, texts_a, cache_dir, batch_size))
        emb_b = _unit(encode_with_cache(backend, texts_b, cache_dir, batch_size))
        sims = emb_a @ emb_b.T
        n = len(texts_a)
        true_mean = float(np.trace(sims) / n)
        random_mean = float((sims.sum() - np.trace(sims)) / (n * n - n))
        rows.append(
            {
                "pivot_language": pivot,
                "other_language": language,
                "n_pairs": n,
                "mean_cosine_true_pairs": true_mean,
                "mean_cosine_mismatched_pairs": random_mean,
                "margin": true_mean - random_mean,
                "top1_retrieval_accuracy": float(np.mean(sims.argmax(axis=1) == np.arange(n))),
                "top1_chance": 1.0 / n,
            }
        )
    return pd.DataFrame(rows)
