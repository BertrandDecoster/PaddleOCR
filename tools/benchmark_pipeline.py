#!/usr/bin/env python3
"""
Benchmark end-to-end OCR pipeline (detection + recognition) against ground truth.

Compares one or more recognition models using the same detection model,
running the full OCR pipeline on uncropped images. Reports exact match
accuracy, normalized edit distance, confidence, and detection miss rate.

Usage:
    python tools/benchmark_pipeline.py \
        --det-model ./exported_ocr_pipeline_v2/models/det_model.onnx \
        --rec-models default=./output/default_server_rec/onnx/model.onnx \
                     finetuned=./output/server_rec_cropped/onnx/model.onnx \
        --dict ./ppocr/utils/dict/ppocrv5_dict.txt \
        --images-dir /path/to/uncropped/images \
        --splits train=train.csv val=val.csv \
        --output ./benchmark_results/pipeline \
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


def normalize_label(text: str) -> str:
    """Normalize a label by stripping non-alphanumeric chars and uppercasing."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def load_ground_truth(csv_path: str) -> list[tuple[str, str]]:
    """Load ground truth from a CSV file (Filename,Label format with header)."""
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize header names to lowercase to handle variants like 'FIlename'
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        for row in reader:
            filename = row.get("filename", "").strip()
            label = row.get("label", "").strip()
            if filename and label:
                entries.append((filename, label))
    return entries


def benchmark_pipeline(
    pipeline,
    ground_truth: list[tuple[str, str]],
    images_dir: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Run the full OCR pipeline on ground truth images.

    For each image:
    - Run detection + recognition
    - If 0 detections: record as detection miss (empty prediction)
    - If 1+ detections: pick result with highest recognition confidence

    Args:
        pipeline: OCRPipeline instance
        ground_truth: List of (filename, label) tuples
        images_dir: Directory containing the images
        max_samples: Limit number of samples

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
            ocr_results, detection_count = pipeline.predict_with_details(image_path)
        except Exception as e:
            print(f"  ERROR on {filename}: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Pick best result by recognition confidence
        if ocr_results:
            best = max(ocr_results, key=lambda r: r["confidence"])
            prediction = best["text"]
            confidence = best["confidence"]
            det_score = best.get("det_score", 0.0)
        else:
            prediction = ""
            confidence = 0.0
            det_score = 0.0

        norm_label = normalize_label(label)
        norm_pred = normalize_label(prediction)

        results.append({
            "filename": filename,
            "label": label,
            "prediction": prediction,
            "confidence": confidence,
            "det_score": det_score,
            "detection_count": detection_count,
            "det_miss": detection_count == 0,
            "raw_match": prediction == label,
            "norm_match": norm_pred == norm_label,
            "norm_label": norm_label,
            "norm_prediction": norm_pred,
        })

        # Progress every 200 images
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} images processed...")

    if skipped:
        print(f"  Skipped {skipped} images (missing or unreadable)")

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-image results."""
    if not results:
        return {"count": 0}

    from rapidfuzz.distance import Levenshtein

    n = len(results)
    raw_matches = sum(1 for r in results if r["raw_match"])
    norm_matches = sum(1 for r in results if r["norm_match"])
    det_misses = sum(1 for r in results if r["det_miss"])
    confidences = [r["confidence"] for r in results]
    det_counts = [r["detection_count"] for r in results]

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
        "raw_ned": float(sum(raw_edit_dists) / n),
        "norm_ned": float(sum(norm_edit_dists) / n),
        "mean_confidence": float(sum(confidences) / n),
        "min_confidence": float(min(confidences)),
        "det_miss_rate": det_misses / n,
        "det_misses": det_misses,
        "avg_det_count": float(sum(det_counts) / n),
    }


def format_table(
    all_metrics: dict[str, dict[str, dict]],
    split_names: list[str],
    model_names: list[str],
) -> str:
    """Format metrics as a readable comparison table."""
    lines = []

    for split in split_names:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"  Split: {split}  (end-to-end pipeline)")
        lines.append(f"{'=' * 72}")

        col_width = 16
        header = f"{'Metric':<28}"
        for model in model_names:
            header += f"{model:>{col_width}}"
        lines.append(header)
        lines.append("-" * (28 + col_width * len(model_names)))

        metrics = all_metrics[split]

        lines.append(f"{'Samples':<28}" + "".join(
            f"{metrics[m]['count']:>{col_width}}" for m in model_names
        ))

        for label, key in [
            ("Raw accuracy", "raw_accuracy"),
            ("Normalized accuracy", "norm_accuracy"),
        ]:
            row = f"{label:<28}"
            for m in model_names:
                row += f"{metrics[m][key]:>{col_width}.4f}"
            lines.append(row)

        for label, key in [
            ("Raw matches", "raw_matches"),
            ("Normalized matches", "norm_matches"),
        ]:
            row = f"{label:<28}"
            for m in model_names:
                row += f"{metrics[m][key]:>{col_width}}"
            lines.append(row)

        for label, key in [
            ("Raw NED (lower=better)", "raw_ned"),
            ("Norm NED (lower=better)", "norm_ned"),
            ("Mean confidence", "mean_confidence"),
            ("Min confidence", "min_confidence"),
            ("Det miss rate", "det_miss_rate"),
            ("Det misses", "det_misses"),
            ("Avg det count", "avg_det_count"),
        ]:
            row = f"{label:<28}"
            for m in model_names:
                val = metrics[m][key]
                if isinstance(val, int):
                    row += f"{val:>{col_width}}"
                else:
                    row += f"{val:>{col_width}.4f}"
            lines.append(row)

    return "\n".join(lines)


def write_per_image_csv(results: list[dict], output_path: str) -> None:
    """Write per-image results to CSV."""
    fieldnames = [
        "Filename", "Label", "Prediction", "Confidence", "DetScore",
        "DetCount", "DetMiss", "RawMatch", "NormMatch",
        "NormLabel", "NormPrediction",
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
                "DetScore": f"{r['det_score']:.4f}",
                "DetCount": r["detection_count"],
                "DetMiss": int(r["det_miss"]),
                "RawMatch": int(r["raw_match"]),
                "NormMatch": int(r["norm_match"]),
                "NormLabel": r["norm_label"],
                "NormPrediction": r["norm_prediction"],
            })


def parse_named_args(values: list[str]) -> dict[str, str]:
    """Parse 'name=value' argument pairs."""
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
        description="Benchmark end-to-end OCR pipeline against ground truth"
    )
    parser.add_argument(
        "--det-model",
        required=True,
        help="Path to detection ONNX model",
    )
    parser.add_argument(
        "--rec-models",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Named recognition ONNX models (e.g., default=model.onnx finetuned=model.onnx)",
    )
    parser.add_argument(
        "--dict",
        required=True,
        help="Path to character dictionary file",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing uncropped images",
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
    parser.add_argument(
        "--det-limit",
        type=int,
        default=960,
        help="Detection max side length (default: 960)",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.3,
        help="Detection threshold (default: 0.3)",
    )
    parser.add_argument(
        "--det-box-thresh",
        type=float,
        default=0.6,
        help="Detection box threshold (default: 0.6)",
    )

    args = parser.parse_args()

    # Parse named arguments
    rec_models = parse_named_args(args.rec_models)
    splits = parse_named_args(args.splits)

    # Validate paths
    if not os.path.isfile(args.det_model):
        parser.error(f"Detection model not found: {args.det_model}")
    for name, path in rec_models.items():
        if not os.path.isfile(path):
            parser.error(f"Recognition model not found: {path}")
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

    # Import OCRPipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from exported_ocr_pipeline_v2.ocr_pipeline import OCRPipeline

    # Build pipelines (one per rec model, sharing det model)
    model_names = list(rec_models.keys())
    pipelines = {}
    for name, rec_path in rec_models.items():
        print(f"\nLoading pipeline with rec model '{name}': {rec_path}")
        pipelines[name] = OCRPipeline(
            det_model_path=args.det_model,
            rec_model_path=rec_path,
            dict_path=args.dict,
            det_limit_side_len=args.det_limit,
            det_thresh=args.det_thresh,
            det_box_thresh=args.det_box_thresh,
            verbose=True,
        )

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
            print(f"\n--- Pipeline benchmark '{model_name}' on '{split_name}' "
                  f"({len(gt)} samples) ---")
            start = time.time()

            results = benchmark_pipeline(
                pipelines[model_name], gt, args.images_dir, args.max_samples
            )
            elapsed = time.time() - start

            metrics = compute_metrics(results)
            all_metrics[split_name][model_name] = metrics
            all_results[split_name][model_name] = results

            if metrics["count"] > 0:
                print(f"  Done in {elapsed:.1f}s "
                      f"({metrics['count']/elapsed:.1f} img/s) - "
                      f"norm_acc={metrics['norm_accuracy']:.4f} "
                      f"det_miss={metrics['det_miss_rate']:.4f}")

    total_elapsed = time.time() - total_start

    # Print comparison table
    table = format_table(all_metrics, split_names, model_names)
    print(table)
    print(f"\nTotal benchmark time: {total_elapsed:.1f}s")

    # Save outputs
    if args.output:
        summary = {
            "det_model": args.det_model,
            "rec_models": {name: str(path) for name, path in rec_models.items()},
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
