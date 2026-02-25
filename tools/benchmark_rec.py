#!/usr/bin/env python3
"""
Benchmark recognition models against ground truth.

Compares one or more ONNX recognition models on labeled datasets,
reporting exact match accuracy, normalized edit distance, and confidence.

Usage:
    python tools/benchmark_rec.py \
        --models default=./output/default_server_rec/onnx/model.onnx \
                 finetuned=./output/server_rec_full/onnx/model.onnx \
        --dict ./ppocr/utils/dict/ppocrv5_dict.txt \
        --images-dir /path/to/images \
        --splits train=train.csv val=val.csv \
        --output ./benchmark_results \
        --per-image
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so we can import from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.export_onnx.infer_onnx_rec import ONNXRecognizer


def normalize_label(text: str) -> str:
    """Normalize a label by stripping non-alphanumeric chars and uppercasing.

    This resolves format differences like dashes, dots, and spaces so that
    'DA-755-RG', 'DA755RG', and 'DA 755 RG' all compare as equal.

    Args:
        text: Raw label text

    Returns:
        Uppercase alphanumeric-only string
    """
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def load_ground_truth(csv_path: str) -> list[tuple[str, str]]:
    """Load ground truth from a CSV file.

    Expected format: Filename,Label (with header row).

    Args:
        csv_path: Path to CSV file

    Returns:
        List of (filename, label) tuples
    """
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["Filename"].strip()
            label = row["Label"].strip()
            if filename and label:
                entries.append((filename, label))
    return entries


def benchmark_model(
    recognizer: ONNXRecognizer,
    ground_truth: list[tuple[str, str]],
    images_dir: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Run a recognition model on all ground truth images.

    Args:
        recognizer: ONNX recognizer instance
        ground_truth: List of (filename, label) tuples
        images_dir: Directory containing the images
        max_samples: Limit number of samples (None for all)

    Returns:
        List of per-image result dicts
    """
    samples = ground_truth[:max_samples] if max_samples else ground_truth
    results = []
    total = len(samples)
    skipped = 0

    for i, (filename, label) in enumerate(samples):
        image_path = os.path.join(images_dir, filename)
        if not os.path.isfile(image_path):
            skipped += 1
            continue

        try:
            prediction, confidence = recognizer.predict(image_path)
        except Exception as e:
            print(f"  ERROR on {filename}: {e}", file=sys.stderr)
            skipped += 1
            continue

        norm_label = normalize_label(label)
        norm_pred = normalize_label(prediction)

        results.append({
            "filename": filename,
            "label": label,
            "prediction": prediction,
            "confidence": confidence,
            "raw_match": prediction == label,
            "norm_match": norm_pred == norm_label,
            "norm_label": norm_label,
            "norm_prediction": norm_pred,
        })

        # Progress every 500 images
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} images processed...")

    if skipped:
        print(f"  Skipped {skipped} images (missing or unreadable)")

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-image results.

    Args:
        results: List of per-image result dicts from benchmark_model()

    Returns:
        Dictionary of computed metrics
    """
    if not results:
        return {"count": 0}

    from rapidfuzz.distance import Levenshtein

    n = len(results)
    raw_matches = sum(1 for r in results if r["raw_match"])
    norm_matches = sum(1 for r in results if r["norm_match"])
    confidences = [r["confidence"] for r in results]

    # Normalized edit distance (0 = identical, 1 = completely different)
    raw_edit_dists = []
    norm_edit_dists = []
    for r in results:
        raw_edit_dists.append(
            Levenshtein.normalized_distance(r["prediction"], r["label"])
        )
        norm_edit_dists.append(
            Levenshtein.normalized_distance(r["norm_prediction"], r["norm_label"])
        )

    return {
        "count": n,
        "raw_accuracy": raw_matches / n,
        "norm_accuracy": norm_matches / n,
        "raw_matches": raw_matches,
        "norm_matches": norm_matches,
        "raw_ned": float(np.mean(raw_edit_dists)),
        "norm_ned": float(np.mean(norm_edit_dists)),
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
    }


def format_table(all_metrics: dict[str, dict[str, dict]], split_names: list[str],
                 model_names: list[str]) -> str:
    """Format metrics as a readable comparison table.

    Args:
        all_metrics: Nested dict of metrics[split][model]
        split_names: Ordered list of split names
        model_names: Ordered list of model names

    Returns:
        Formatted table string
    """
    lines = []

    for split in split_names:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"  Split: {split}")
        lines.append(f"{'=' * 72}")

        # Header
        col_width = 16
        header = f"{'Metric':<28}"
        for model in model_names:
            header += f"{model:>{col_width}}"
        lines.append(header)
        lines.append("-" * (28 + col_width * len(model_names)))

        metrics = all_metrics[split]
        count = metrics[model_names[0]]["count"]
        lines.append(f"{'Samples':<28}" + "".join(
            f"{metrics[m]['count']:>{col_width}}" for m in model_names
        ))

        # Accuracy rows
        for label, key in [
            ("Raw accuracy", "raw_accuracy"),
            ("Normalized accuracy", "norm_accuracy"),
        ]:
            row = f"{label:<28}"
            for m in model_names:
                val = metrics[m][key]
                row += f"{val:>{col_width}.4f}"
            lines.append(row)

        # Match count rows
        for label, key in [
            ("Raw matches", "raw_matches"),
            ("Normalized matches", "norm_matches"),
        ]:
            row = f"{label:<28}"
            for m in model_names:
                val = metrics[m][key]
                row += f"{val:>{col_width}}"
            lines.append(row)

        # Distance / confidence rows
        for label, key in [
            ("Raw NED (lower=better)", "raw_ned"),
            ("Norm NED (lower=better)", "norm_ned"),
            ("Mean confidence", "mean_confidence"),
            ("Min confidence", "min_confidence"),
        ]:
            row = f"{label:<28}"
            for m in model_names:
                val = metrics[m][key]
                row += f"{val:>{col_width}.4f}"
            lines.append(row)

    return "\n".join(lines)


def write_per_image_csv(results: list[dict], output_path: str) -> None:
    """Write per-image results to CSV.

    Args:
        results: Per-image result dicts
        output_path: Path for output CSV file
    """
    fieldnames = [
        "Filename", "Label", "Prediction", "Confidence",
        "RawMatch", "NormMatch", "NormLabel", "NormPrediction",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Filename": r["filename"],
                "Label": r["label"],
                "Prediction": r["prediction"],
                "Confidence": f"{r['confidence']:.4f}",
                "RawMatch": int(r["raw_match"]),
                "NormMatch": int(r["norm_match"]),
                "NormLabel": r["norm_label"],
                "NormPrediction": r["norm_prediction"],
            })


def parse_named_args(values: list[str]) -> dict[str, str]:
    """Parse 'name=value' argument pairs.

    Args:
        values: List of 'name=value' strings

    Returns:
        Dict mapping names to values
    """
    result = {}
    for v in values:
        if "=" not in v:
            raise argparse.ArgumentTypeError(
                f"Expected 'name=path' format, got: {v}"
            )
        name, path = v.split("=", 1)
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark recognition models against ground truth"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Named ONNX models to compare (e.g., default=model.onnx finetuned=model.onnx)",
    )
    parser.add_argument(
        "--dict",
        required=True,
        help="Path to character dictionary file",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing images referenced by CSV files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Named CSV splits (e.g., train=train.csv val=val.csv)",
    )
    parser.add_argument(
        "--output",
        help="Output directory for JSON summary and per-image CSVs",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Write per-image prediction CSVs for error analysis",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per split (for quick testing)",
    )

    args = parser.parse_args()

    # Parse named arguments
    models = parse_named_args(args.models)
    splits = parse_named_args(args.splits)

    # Validate paths
    for name, path in models.items():
        if not os.path.isfile(path):
            parser.error(f"Model file not found: {path}")
    for name, path in splits.items():
        if not os.path.isfile(path):
            parser.error(f"CSV file not found: {path}")
    if not os.path.isdir(args.images_dir):
        parser.error(f"Images directory not found: {args.images_dir}")
    if not os.path.isfile(args.dict):
        parser.error(f"Dictionary file not found: {args.dict}")

    # Create output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Load models
    model_names = list(models.keys())
    recognizers = {}
    for name, path in models.items():
        print(f"\nLoading model '{name}': {path}")
        recognizers[name] = ONNXRecognizer(path, args.dict, verbose=True)
        print()

    # Load ground truth for each split
    split_names = list(splits.keys())
    ground_truths = {}
    for name, path in splits.items():
        gt = load_ground_truth(path)
        ground_truths[name] = gt
        print(f"Split '{name}': {len(gt)} samples from {path}")

    # Run benchmarks
    all_metrics: dict[str, dict[str, dict]] = {}
    all_results: dict[str, dict[str, list[dict]]] = {}

    total_start = time.time()

    for split_name in split_names:
        gt = ground_truths[split_name]
        all_metrics[split_name] = {}
        all_results[split_name] = {}

        for model_name in model_names:
            print(f"\n--- Benchmarking '{model_name}' on '{split_name}' "
                  f"({len(gt)} samples) ---")
            start = time.time()

            results = benchmark_model(
                recognizers[model_name], gt, args.images_dir, args.max_samples
            )
            elapsed = time.time() - start

            metrics = compute_metrics(results)
            all_metrics[split_name][model_name] = metrics
            all_results[split_name][model_name] = results

            print(f"  Done in {elapsed:.1f}s "
                  f"({len(results)/elapsed:.1f} img/s) - "
                  f"norm_acc={metrics['norm_accuracy']:.4f}")

    total_elapsed = time.time() - total_start

    # Print comparison table
    table = format_table(all_metrics, split_names, model_names)
    print(table)
    print(f"\nTotal benchmark time: {total_elapsed:.1f}s")

    # Save outputs
    if args.output:
        # JSON summary
        summary = {
            "models": {name: str(path) for name, path in models.items()},
            "splits": {name: str(path) for name, path in splits.items()},
            "images_dir": args.images_dir,
            "max_samples": args.max_samples,
            "metrics": all_metrics,
            "total_time_seconds": total_elapsed,
        }
        summary_path = os.path.join(args.output, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        # Per-image CSVs
        if args.per_image:
            for split_name in split_names:
                for model_name in model_names:
                    csv_path = os.path.join(
                        args.output, f"{model_name}_{split_name}.csv"
                    )
                    write_per_image_csv(
                        all_results[split_name][model_name], csv_path
                    )
                    print(f"Per-image results: {csv_path}")


if __name__ == "__main__":
    main()
