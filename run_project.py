"""Run the multilingual fake-news detection experiment from the project root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fakenews_xling.data import (
    FAKECOVID_FILENAME, FAKECOVID_RECORD, download_fakecovid, generate_synthetic_demo,
)
from fakenews_xling.embeddings import DEFAULT_ENCODER
from fakenews_xling.experiment import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Multilingual fake-news detection with cross-lingual Transformer sentence embeddings."
    )
    p.add_argument("--source", choices=["real", "demo"], default="real",
                   help="real downloads FakeCovid (check its Zenodo record for the licence) if missing; demo uses generated synthetic data.")
    p.add_argument("--data", type=str, default=None, help="Path to your own CSV (overrides --source).")
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--language-col", default=None)
    p.add_argument("--date-col", default=None)
    p.add_argument("--group-col", default=None, help="Column whose rows are related (e.g. translations of one claim).")
    p.add_argument("--label-map", default=None,
                   help='JSON file mapping raw labels to 0 (real), 1 (fake) or null (drop), e.g. {"False": 1, "True": 0}.')
    p.add_argument("--date-dayfirst", action="store_true", help="Parse dates such as 04/01/2020 as day/month.")
    p.add_argument("--parallel-csv", default=None, help="Optional CSV with columns text, language, claim_id for the alignment analysis.")
    p.add_argument("--encoder", default=None,
                   help=f"Sentence-encoder name or 'hashing' (offline test only). Default: {DEFAULT_ENCODER} (real), hashing (demo).")
    p.add_argument("--split", choices=["auto", "grouped", "chronological", "stratified"], default="auto")
    p.add_argument("--transfer-source", default=None, help="Language to train on for the transfer test (default: en).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output folder (default: outputs, or outputs_demo for the synthetic demo).")
    args = p.parse_args()
    if args.output is None:
        demo = args.source == "demo" and not args.data
        args.output = str(ROOT / ("outputs_demo" if demo else "outputs"))
    return args


def main():
    args = parse_args()
    dataset_name = "custom"
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
        dataset_name = data_path.stem
    elif args.source == "demo":
        data_path = DATA_DIR / "demo_synthetic_multilingual.csv"
        dataset_name = "synthetic_demo"
        if not data_path.exists():
            print("Creating clearly labelled synthetic demonstration data...")
            generate_synthetic_demo(data_path)
    else:
        data_path = DATA_DIR / "real" / FAKECOVID_FILENAME
        dataset_name = "fakecovid"
        if not data_path.exists():
            print("Downloading the FakeCovid dataset from Zenodo (about 55 MB)...")
            try:
                download_fakecovid(data_path)
            except Exception as exc:  # noqa: BLE001
                print("\nCould not download the dataset:", exc)
                print(f"\nManual option: open {FAKECOVID_RECORD}, download {FAKECOVID_FILENAME} and save it as:")
                print(f"  {data_path}")
                print("Then re-run the same command. Offline software check (synthetic data):")
                print("  python run_project.py --source demo\n")
                return 2

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    print(f"Using data file: {data_path}")
    if dataset_name == "synthetic_demo":
        print("DATA WARNING: synthetic demonstration data; these numbers are not research results.")

    encoder = args.encoder or ("hashing" if dataset_name == "synthetic_demo" else DEFAULT_ENCODER)
    if encoder == "hashing":
        print("ENCODER WARNING: 'hashing' is an offline smoke-test encoder and is not cross-lingual.")

    label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if args.label_map else None
    columns = {
        "text_col": args.text_col, "label_col": args.label_col, "language_col": args.language_col,
        "date_col": args.date_col, "group_col": args.group_col,
    }
    config = ExperimentConfig(
        seed=args.seed, encoder=encoder, batch_size=args.batch_size, split=args.split,
        transfer_source=args.transfer_source, dayfirst=args.date_dayfirst,
    )
    metrics, meta = run_experiment(
        data_path=data_path, output_dir=Path(args.output).expanduser().resolve(), config=config,
        dataset_name=dataset_name, columns=columns, label_map=label_map,
        parallel_path=args.parallel_csv, cache_dir=DATA_DIR / "cache",
    )

    loading = meta["loading"]
    print("\nColumns used:", loading["columns_used"])
    print("Raw label values (top):", loading["raw_label_counts"])
    print("Class counts after cleaning:", loading["label_counts"])
    print("Language counts:", loading["language_counts"])
    print(f"Split strategy: {meta['split']['strategy']} -> {meta['split']['counts']}")
    for note in meta["split"]["notes"]:
        print("NOTE:", note)
    print("\nTEST-SET MODEL COMPARISON")
    cols = ["Model", "Macro_F1", "Fake_F1", "Fake_precision", "Fake_recall", "Accuracy"]
    print(metrics[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nCompleted. Outputs saved to: {Path(args.output).resolve()}")
    print("Check the printed raw labels against the class counts, then read outputs/report_results.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
