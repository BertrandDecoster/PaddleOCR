#!/usr/bin/env python3
"""
Data preparation script for PaddleOCR recognition finetuning with pre-split CSVs.

Handles pre-split train/val ground truth files, validates images, and fixes
data leakage issues (duplicates within/across splits).
"""

import argparse
import csv
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare pre-split data for PaddleOCR recognition finetuning"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training ground truth CSV file (Filename,Label format)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Validation ground truth CSV file (Filename,Label format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of using symlinks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any images are missing (default: warn only)",
    )
    return parser.parse_args()


def read_csv_ground_truth(csv_path):
    """
    Read ground truth from CSV file with Filename,Label format.

    Returns list of (filename, label) tuples preserving order.
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("Filename") or row.get("filename")
            label = row.get("Label") or row.get("label")
            if filename and label:
                data.append((filename.strip(), label.strip()))
    return data


def find_duplicates(data):
    """
    Find duplicate filenames within a dataset.

    Returns:
        duplicates: list of filenames that appear more than once
        first_occurrence: dict mapping filename to its first (filename, label) tuple
    """
    filename_counts = Counter(filename for filename, _ in data)
    duplicates = [f for f, count in filename_counts.items() if count > 1]

    # Track first occurrence for deduplication
    first_occurrence = {}
    for filename, label in data:
        if filename not in first_occurrence:
            first_occurrence[filename] = (filename, label)

    return duplicates, first_occurrence


def dedupe_data(data, source_name="dataset"):
    """
    Remove duplicate entries, keeping first occurrence.

    Returns:
        deduped_data: list with duplicates removed
        removed_duplicates: list of removed (filename, label) tuples
    """
    duplicates, first_occurrence = find_duplicates(data)

    if duplicates:
        print(f"  Found {len(duplicates)} duplicate filename(s) in {source_name}:")
        for dup in duplicates[:5]:
            print(f"    - {dup}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")

    # Build deduped list preserving order (only first occurrence)
    seen = set()
    deduped_data = []
    removed_duplicates = []

    for filename, label in data:
        if filename not in seen:
            seen.add(filename)
            deduped_data.append((filename, label))
        else:
            removed_duplicates.append((filename, label))

    return deduped_data, removed_duplicates


def find_cross_contamination(train_data, val_data):
    """
    Find filenames that appear in both train and val sets.

    Returns set of filenames that are in both splits.
    """
    train_filenames = {f for f, _ in train_data}
    val_filenames = {f for f, _ in val_data}
    return train_filenames & val_filenames


def validate_images(data, images_dir, source_name="dataset"):
    """
    Validate that all images exist.

    Returns:
        valid_data: list of (filename, label) for existing images
        missing: list of filenames that don't exist
    """
    valid_data = []
    missing = []

    for filename, label in data:
        img_path = images_dir / filename
        if img_path.exists():
            valid_data.append((filename, label))
        else:
            missing.append(filename)

    if missing:
        print(f"  {len(missing)} missing image(s) in {source_name}:")
        for f in missing[:5]:
            print(f"    - {f}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    return valid_data, missing


def write_cleaning_report(output_dir, stats):
    """Write a detailed cleaning report."""
    report_path = output_dir / "cleaning_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Data Preparation Cleaning Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("INPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Images directory: {stats['images_dir']}\n")
        f.write(f"Train CSV: {stats['train_csv']}\n")
        f.write(f"Val CSV: {stats['val_csv']}\n\n")

        f.write("ORIGINAL COUNTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train rows: {stats['original_train_count']}\n")
        f.write(f"Val rows: {stats['original_val_count']}\n\n")

        f.write("DEDUPLICATION (within splits)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train duplicates removed: {stats['train_duplicates_removed']}\n")
        if stats['train_duplicate_files']:
            for dup in stats['train_duplicate_files']:
                f.write(f"  - {dup}\n")
        f.write(f"Val duplicates removed: {stats['val_duplicates_removed']}\n")
        if stats['val_duplicate_files']:
            for dup in stats['val_duplicate_files']:
                f.write(f"  - {dup}\n")
        f.write("\n")

        f.write("CROSS-CONTAMINATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Files in both train and val: {stats['cross_contamination_count']}\n")
        if stats['cross_contamination_files']:
            f.write("(Removed from val, kept in train):\n")
            for dup in stats['cross_contamination_files']:
                f.write(f"  - {dup}\n")
        f.write("\n")

        f.write("MISSING IMAGES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train missing: {stats['train_missing_count']}\n")
        f.write(f"Val missing: {stats['val_missing_count']}\n\n")

        f.write("FINAL COUNTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train samples: {stats['final_train_count']}\n")
        f.write(f"Val samples: {stats['final_val_count']}\n")
        f.write(f"Total samples: {stats['final_train_count'] + stats['final_val_count']}\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train list: {stats['train_list_path']}\n")
        f.write(f"Val list: {stats['val_list_path']}\n")
        f.write(f"Images directory: {stats['output_images_dir']}\n")

    return report_path


def prepare_data_presplit(
    images_dir, train_csv, val_csv, output_dir, copy_images=False, strict=False
):
    """
    Prepare pre-split data for PaddleOCR recognition training.

    Args:
        images_dir: Directory containing source images
        train_csv: Training CSV file with Filename,Label columns
        val_csv: Validation CSV file with Filename,Label columns
        output_dir: Output directory for prepared data
        copy_images: If True, copy images; otherwise create symlinks
        strict: If True, fail on missing images; otherwise warn

    Returns:
        dict with paths to created files and statistics
    """
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Create output directories
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Read ground truth files
    print("\nReading ground truth files...")
    train_data = read_csv_ground_truth(train_csv)
    val_data = read_csv_ground_truth(val_csv)

    if not train_data:
        raise ValueError(f"No data found in {train_csv}")
    if not val_data:
        raise ValueError(f"No data found in {val_csv}")

    original_train_count = len(train_data)
    original_val_count = len(val_data)
    print(f"  Train CSV: {original_train_count} rows")
    print(f"  Val CSV: {original_val_count} rows")

    # Step 1: Dedupe within each split
    print("\nDeduplicating within splits...")
    train_data, train_removed = dedupe_data(train_data, "train")
    val_data, val_removed = dedupe_data(val_data, "val")

    train_duplicates = find_duplicates(read_csv_ground_truth(train_csv))[0]
    val_duplicates = find_duplicates(read_csv_ground_truth(val_csv))[0]

    # Step 2: Handle cross-contamination
    print("\nChecking for cross-contamination...")
    cross_contamination = find_cross_contamination(train_data, val_data)

    if cross_contamination:
        print(f"  Found {len(cross_contamination)} file(s) in both train and val:")
        for f in list(cross_contamination)[:5]:
            print(f"    - {f}")
        if len(cross_contamination) > 5:
            print(f"    ... and {len(cross_contamination) - 5} more")
        print("  Removing from val (keeping in train)...")

        # Remove cross-contaminated files from val
        val_data = [(f, l) for f, l in val_data if f not in cross_contamination]
    else:
        print("  No cross-contamination found (clean split)")

    # Step 3: Validate images exist
    print("\nValidating images exist...")
    train_data, train_missing = validate_images(train_data, images_dir, "train")
    val_data, val_missing = validate_images(val_data, images_dir, "val")

    if strict and (train_missing or val_missing):
        raise ValueError(
            f"Missing images in strict mode: {len(train_missing)} train, {len(val_missing)} val"
        )

    if not train_data:
        raise ValueError("No valid training samples after cleaning")
    if not val_data:
        raise ValueError("No valid validation samples after cleaning")

    # Step 4: Create image links/copies and write label files
    print("\nCreating output files...")

    def process_split(split_data, label_file_path):
        with open(label_file_path, "w", encoding="utf-8") as f:
            for filename, label in split_data:
                src_path = images_dir / filename
                dst_path = output_images_dir / filename

                # Copy or symlink image
                if not dst_path.exists():
                    if copy_images:
                        shutil.copy2(src_path, dst_path)
                    else:
                        dst_path.symlink_to(src_path)

                # Write label line: relative_path\tlabel
                rel_path = f"images/{filename}"
                f.write(f"{rel_path}\t{label}\n")

    train_list_path = output_dir / "train_list.txt"
    val_list_path = output_dir / "val_list.txt"

    process_split(train_data, train_list_path)
    process_split(val_data, val_list_path)

    # Compile statistics
    stats = {
        "images_dir": str(images_dir),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "original_train_count": original_train_count,
        "original_val_count": original_val_count,
        "train_duplicates_removed": len(train_removed),
        "train_duplicate_files": train_duplicates,
        "val_duplicates_removed": len(val_removed),
        "val_duplicate_files": val_duplicates,
        "cross_contamination_count": len(cross_contamination),
        "cross_contamination_files": list(cross_contamination),
        "train_missing_count": len(train_missing),
        "val_missing_count": len(val_missing),
        "final_train_count": len(train_data),
        "final_val_count": len(val_data),
        "output_dir": str(output_dir),
        "output_images_dir": str(output_images_dir),
        "train_list_path": str(train_list_path),
        "val_list_path": str(val_list_path),
    }

    # Write cleaning report
    report_path = write_cleaning_report(output_dir, stats)
    stats["report_path"] = str(report_path)

    return stats


def main():
    args = parse_args()

    print("=" * 60)
    print("Preparing pre-split data for PaddleOCR recognition finetuning")
    print("=" * 60)
    print(f"Images dir: {args.images_dir}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"Copy images: {args.copy_images}")
    print(f"Strict mode: {args.strict}")

    stats = prepare_data_presplit(
        images_dir=args.images_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        copy_images=args.copy_images,
        strict=args.strict,
    )

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nCleaning summary:")
    print(f"  Train duplicates removed: {stats['train_duplicates_removed']}")
    print(f"  Val duplicates removed: {stats['val_duplicates_removed']}")
    print(f"  Cross-contamination removed: {stats['cross_contamination_count']}")
    print(f"  Train missing images: {stats['train_missing_count']}")
    print(f"  Val missing images: {stats['val_missing_count']}")

    print(f"\nFinal counts:")
    print(f"  Train samples: {stats['final_train_count']}")
    print(f"  Val samples: {stats['final_val_count']}")
    print(f"  Total: {stats['final_train_count'] + stats['final_val_count']}")

    print(f"\nOutput files:")
    print(f"  {stats['train_list_path']}")
    print(f"  {stats['val_list_path']}")
    print(f"  {stats['report_path']}")


if __name__ == "__main__":
    main()
