"""Classifier heads and traditional baselines, all tuned on the validation split only."""
from __future__ import annotations

import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.1, 1.0, 10.0)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Python's default token pattern (\w\w+) splits Devanagari words at vowel signs
# (combining marks are not "word" characters), which silently ruins word-level
# TF-IDF for Indian-language text. Splitting on whitespace/punctuation avoids that.
_TOKEN = re.compile(r"[^\s.,;:!?()\[\]{}\"'“”‘’।|/\\\-–—]+")


def multilingual_tokenizer(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def select_threshold(y_true, proba) -> tuple[float, float]:
    """Pick the decision threshold with the best validation macro-F1 (ties -> closest to 0.5)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best_threshold, best_score = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        score = f1_score(y_true, (proba >= threshold).astype(int), average="macro", zero_division=0)
        better = score > best_score + 1e-12
        tie_closer = abs(score - best_score) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        if better or tie_closer:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def _select_logreg(X_train, y_train, X_val, y_val, seed: int):
    best = None
    for c_value in C_GRID:
        clf = LogisticRegression(
            C=c_value, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed
        ).fit(X_train, y_train)
        threshold, score = select_threshold(y_val, clf.predict_proba(X_val)[:, 1])
        if best is None or score > best[0] + 1e-12:
            best = (score, c_value, threshold, clf)
    score, c_value, threshold, clf = best
    return clf, {"C": c_value, "threshold": threshold, "val_macro_f1": score}


def _vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(
            tokenizer=multilingual_tokenizer, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True,
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000,
            sublinear_tf=True, lowercase=True,
        )
    raise ValueError(f"Unknown TF-IDF kind '{kind}'.")


def train_text_model(kind: str, train_texts, y_train, val_texts, y_val, seed: int):
    """TF-IDF (word or character n-grams) + logistic regression."""
    vectorizer = _vectorizer(kind)
    X_train = vectorizer.fit_transform(list(train_texts))
    X_val = vectorizer.transform(list(val_texts))
    clf, meta = _select_logreg(X_train, y_train, X_val, y_val, seed)
    meta["n_features"] = int(X_train.shape[1])
    return Pipeline([("tfidf", vectorizer), ("clf", clf)]), meta


def train_embedding_model(kind: str, E_train, y_train, E_val, y_val, seed: int):
    """Classifier head on frozen sentence embeddings (scaler fitted on training rows only)."""
    scaler = StandardScaler().fit(E_train)
    X_train, X_val = scaler.transform(E_train), scaler.transform(E_val)
    if kind == "logreg":
        clf, meta = _select_logreg(X_train, y_train, X_val, y_val, seed)
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, max_iter=400, random_state=seed,
        ).fit(X_train, y_train)
        threshold, score = select_threshold(y_val, clf.predict_proba(X_val)[:, 1])
        meta = {"threshold": threshold, "val_macro_f1": score, "iterations": int(clf.n_iter_)}
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=20, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=seed,
        ).fit(X_train, y_train)
        threshold, score = select_threshold(y_val, clf.predict_proba(X_val)[:, 1])
        meta = {"threshold": threshold, "val_macro_f1": score}
    else:
        raise ValueError(f"Unknown embedding head '{kind}'.")
    return Pipeline([("scaler", scaler), ("clf", clf)]), meta
