"""Text encoders: a multilingual Transformer sentence encoder and an offline smoke-test backend."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HashingBackend:
    """Character n-gram hashing. NOT cross-lingual: used only to test the software offline."""

    cross_lingual_status = "no (offline smoke-test backend; character n-grams are not shared across scripts)"

    def __init__(self, n_features: int = 512):
        self.name = f"hashing-char-ngrams-{n_features}"
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=n_features,
            alternate_sign=False, norm="l2", lowercase=True,
        )

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        return self._vectorizer.transform(list(texts)).toarray().astype(np.float32)


class SentenceTransformerBackend:
    """Pretrained Transformer sentence encoder (loaded lazily from the Hugging Face hub)."""

    cross_lingual_status = (
        "depends on the chosen model - check its model card for the languages it covers "
        "(Hindi and other Indian languages in particular)"
    )

    def __init__(self, model_name: str = DEFAULT_ENCODER):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "The 'sentence-transformers' package is not installed. Run setup_windows.bat "
                    "(or: pip install -r requirements.txt). For an offline software check only, "
                    "use: python run_project.py --source demo"
                ) from exc
            self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        model = self._load()
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=len(texts) > 500,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def get_encoder(spec: str):
    if spec.strip().lower() in {"hashing", "hash", "offline"}:
        return HashingBackend()
    return SentenceTransformerBackend(spec.strip())


def encode_with_cache(backend, texts: list[str], cache_dir: str | Path, batch_size: int = 64) -> np.ndarray:
    """Encode texts once and reuse the result (keyed by encoder name and exact text list)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1((backend.name + "\n" + "\x1f".join(texts)).encode("utf-8")).hexdigest()[:20]
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", backend.name).strip("_")[:40]
    path = cache_dir / f"emb_{safe_name}_{digest}.npy"
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    embeddings = backend.encode(texts, batch_size=batch_size)
    np.save(path, embeddings)
    return embeddings
