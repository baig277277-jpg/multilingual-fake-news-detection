"""Automated checks for labels, cleaning, leakage-safe splitting, alignment and the pipeline."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews_xling.alignment import alignment_report
from fakenews_xling.data import (
    DEMO_LANGUAGES, generate_synthetic_demo, load_dataset_csv, map_labels, normalize_label,
    preprocess_text, split_frame,
)
from fakenews_xling.embeddings import HashingBackend, encode_with_cache
from fakenews_xling.experiment import ExperimentConfig, run_experiment
from fakenews_xling.explain import build_explanations, build_llm_prompt
from fakenews_xling.models import multilingual_tokenizer, select_threshold


def _write_csv(directory: Path, rows: list[dict], name: str = "data.csv") -> Path:
    path = directory / name
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


class LabelTests(unittest.TestCase):
    def test_fact_check_verdicts(self):
        expected = {
            "False": 1, "TRUE": 0, "Mostly false": 1, "not true": 1, "Untrue": 1, "incorrect": 1,
            "Misleading": 1, "Fake News": 1, "real": 0, "Correct": 0, "1": 1, "0": 0,
            "Partly False": None, "Half True": None, "satire": None, "unverified": None,
            "": None, "something else": None,
        }
        for raw, label in expected.items():
            self.assertEqual(normalize_label(raw), label, msg=raw)
        self.assertIsNone(normalize_label(np.nan))

    def test_explicit_label_map_drops_unmapped_values(self):
        mapped = map_labels(pd.Series(["yes", "no", "maybe"]), {"yes": 1, "no": 0})
        self.assertEqual(mapped.iloc[:2].tolist(), [1.0, 0.0])
        self.assertTrue(np.isnan(mapped.iloc[2]))


class TextTests(unittest.TestCase):
    def test_preprocess_replaces_urls_normalises_unicode_and_whitespace(self):
        cleaned = preprocess_text("  ＡＢＣ   see https://example.com/x?y=1  now ")
        self.assertEqual(cleaned, "ABC see <URL> now")
        self.assertEqual(preprocess_text(None), "")

    def test_tokenizer_keeps_devanagari_words_intact(self):
        self.assertEqual(
            multilingual_tokenizer("शहर की पानी की आपूर्ति!"),
            ["शहर", "की", "पानी", "की", "आपूर्ति"],
        )
        # The default scikit-learn token pattern breaks such words at vowel signs.
        default_tokens = re.findall(r"(?u)\b\w\w+\b", "आपूर्ति")
        self.assertNotEqual(default_tokens, ["आपूर्ति"])


class LoadingTests(unittest.TestCase):
    def test_duplicates_conflicts_and_language_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"text": "The same claim repeated here", "label": "False", "language": "English"},
                {"text": "The same claim repeated here", "label": "False", "language": "English"},
                {"text": "Conflicting duplicate text", "label": "False", "language": "English"},
                {"text": "Conflicting duplicate text", "label": "True", "language": "English"},
                {"text": "A different true statement", "label": "True", "language": "Hindi"},
                {"text": "short", "label": "True", "language": "English"},
                {"text": "Ambiguous rating example", "label": "Half True", "language": "English"},
            ]
            frame, info = load_dataset_csv(_write_csv(Path(tmp), rows))
            self.assertEqual(sorted(frame["text"]), ["A different true statement", "The same claim repeated here"])
            self.assertEqual(set(frame["language"]), {"en", "hi"})
            self.assertEqual(info["dropped_conflicting_duplicate_texts"], 2)
            self.assertEqual(info["dropped_exact_duplicate_texts"], 1)
            self.assertEqual(info["dropped_short_or_empty_text"], 1)
            self.assertEqual(info["dropped_unmapped_or_ambiguous_label"], 1)

    def test_missing_label_column_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [{"text": "Some long enough text", "other": 1}])
            with self.assertRaisesRegex(ValueError, "label"):
                load_dataset_csv(path)


class SplitTests(unittest.TestCase):
    def _demo_frame(self, tmp):
        path = generate_synthetic_demo(Path(tmp) / "demo.csv")
        frame, _ = load_dataset_csv(path)
        return frame

    def test_demo_is_parallel_with_consistent_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._demo_frame(tmp)
            self.assertEqual(len(frame), 24 * 6 * 3)
            self.assertTrue((frame.groupby("claim_id")["language"].nunique() == len(DEMO_LANGUAGES)).all())
            self.assertTrue((frame.groupby("claim_id")["label"].nunique() == 1).all())

    def test_grouped_split_never_shares_a_claim_between_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._demo_frame(tmp)
            train, val, test, info = split_frame(frame, "auto", seed=7)
            self.assertEqual(info["strategy"], "grouped")
            ids = [set(part["claim_id"]) for part in (train, val, test)]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            self.assertEqual(len(train) + len(val) + len(test), len(frame))
            self.assertAlmostEqual(len(train) / len(frame), 0.70, delta=0.08)
            for part in (train, val, test):
                self.assertEqual(part["label"].nunique(), 2)

    def test_chronological_split_orders_time_and_scaler_texts_do_not_overlap(self):
        rng = np.random.default_rng(0)
        n = 400
        frame = pd.DataFrame(
            {
                "text": [f"statement number {i} about topic {i % 17}" for i in range(n)],
                "label": rng.integers(0, 2, n),
                "language": "en",
                "date": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")[rng.permutation(n)],
            }
        )
        train, val, test, info = split_frame(frame, "auto")
        self.assertEqual(info["strategy"], "chronological")
        self.assertLessEqual(train["date"].max(), val["date"].min())
        self.assertLessEqual(val["date"].max(), test["date"].min())
        self.assertFalse(set(train["text"]) & set(test["text"]))

    def test_random_fallback_warns_and_too_few_rows_fail(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {"text": [f"text {i}" for i in range(400)], "label": rng.integers(0, 2, 400), "language": "en"}
        )
        _, _, _, info = split_frame(frame, "auto")
        self.assertEqual(info["strategy"], "stratified")
        self.assertTrue(info["notes"])
        with self.assertRaises(ValueError):
            split_frame(frame.iloc[:100], "auto")


class ModelHelperTests(unittest.TestCase):
    def test_threshold_selection_on_separable_scores(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        threshold, score = select_threshold(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
        self.assertEqual(score, 1.0)
        self.assertTrue(0.3 < threshold <= 0.7)
        select_threshold(np.zeros(5, dtype=int), np.linspace(0, 1, 5))  # single class must not crash

    def test_hashing_backend_is_deterministic_and_normalised(self):
        backend = HashingBackend()
        first = backend.encode(["some text here", "कुछ पाठ यहाँ"])
        second = backend.encode(["some text here", "कुछ पाठ यहाँ"])
        self.assertEqual(first.shape, (2, 512))
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)


class _ToyBackend:
    """Maps 'claim<k> <lang>' to a one-hot vector of k, i.e. a perfectly aligned encoder."""
    name = "toy-perfect-alignment"
    cross_lingual_status = "toy"

    def encode(self, texts, batch_size=64):
        out = np.zeros((len(texts), 30), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, int(re.search(r"claim(\d+)", text).group(1))] = 1.0
        return out


class AlignmentAndExplanationTests(unittest.TestCase):
    def test_perfect_alignment_scores_one_and_mismatched_scores_zero(self):
        rows = [
            {"claim_id": f"c{k}", "language": lang, "text": f"claim{k} {lang}"}
            for k in range(20) for lang in ("en", "hi")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = alignment_report(_ToyBackend(), pd.DataFrame(rows), "en", tmp)
        self.assertEqual(len(report), 1)
        row = report.iloc[0]
        self.assertAlmostEqual(row["mean_cosine_true_pairs"], 1.0)
        self.assertAlmostEqual(row["mean_cosine_mismatched_pairs"], 0.0)
        self.assertAlmostEqual(row["top1_retrieval_accuracy"], 1.0)

    def test_embedding_cache_reuses_and_matches(self):
        backend = HashingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            first = encode_with_cache(backend, ["alpha beta", "gamma delta"], tmp)
            second = encode_with_cache(backend, ["alpha beta", "gamma delta"], tmp)
            self.assertEqual(len(list(Path(tmp).glob("*.npy"))), 1)
            np.testing.assert_allclose(first, second)

    def test_explanations_only_use_training_claims_and_prompt_forbids_verdicts(self):
        rng = np.random.default_rng(3)
        train = pd.DataFrame(
            {"text": [f"train claim {i}" for i in range(30)], "language": "en", "label": rng.integers(0, 2, 30)}
        )
        test = pd.DataFrame(
            {"text": [f"test claim {i}" for i in range(12)], "language": "hi", "label": [0, 1] * 6}
        )
        E_train, E_test = rng.normal(size=(30, 8)), rng.normal(size=(12, 8))
        proba = rng.random(12)
        items = build_explanations(test, train, E_test, E_train, proba, 0.5)
        self.assertTrue(items)
        train_texts = set(train["text"])
        for item in items:
            for neighbour in item["neighbours"]:
                self.assertIn(neighbour["text"], train_texts)
            self.assertIn("not evidence", item["explanation"])
        prompt = build_llm_prompt("a claim", "en", "Real", 0.2, items[0]["neighbours"])
        self.assertIn("Do not say whether the claim is true or false", prompt)


class EndToEndTests(unittest.TestCase):
    def test_demo_run_writes_outputs_and_test_labels_do_not_influence_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = generate_synthetic_demo(tmp / "demo.csv")
            config = ExperimentConfig(encoder="hashing")
            metrics, meta = run_experiment(
                csv_path, tmp / "out1", config, dataset_name="synthetic_demo", cache_dir=tmp / "cache"
            )
            self.assertEqual(len(metrics), 6)
            self.assertEqual(meta["dataset_type"], "synthetic_demo")
            self.assertEqual(meta["split"]["strategy"], "grouped")
            for name in ("metrics.csv", "report_results.md", "test_predictions.csv", "explanations.md",
                         "transfer_metrics.csv", "alignment.csv", "model/metadata.json"):
                self.assertTrue((tmp / "out1" / name).exists(), msg=name)
            for name in ("model_comparison.png", "confusion_matrix.png", "per_language_f1.png"):
                self.assertTrue((tmp / "out1" / "plots" / name).exists(), msg=name)

            # Flip the label of every claim that ended up in the test split, then rerun.
            first = pd.read_csv(tmp / "out1" / "test_predictions.csv")
            raw = pd.read_csv(csv_path)
            flip = raw["claim_id"].isin(set(first["claim_id"]))
            raw.loc[flip, "label"] = raw.loc[flip, "label"].map({"fake": "real", "real": "fake"})
            flipped_path = tmp / "flipped.csv"
            raw.to_csv(flipped_path, index=False, encoding="utf-8")
            run_experiment(flipped_path, tmp / "out2", config, dataset_name="synthetic_demo", cache_dir=tmp / "cache")
            second = pd.read_csv(tmp / "out2" / "test_predictions.csv")

            self.assertEqual(first["claim_id"].tolist(), second["claim_id"].tolist())
            score_columns = [c for c in first.columns if c.endswith("_score")]
            self.assertTrue(score_columns)
            for column in score_columns:
                np.testing.assert_allclose(first[column], second[column], err_msg=column)
            self.assertFalse((first["true_label"] == second["true_label"]).any())


if __name__ == "__main__":
    unittest.main()
