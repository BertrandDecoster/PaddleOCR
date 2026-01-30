#!/usr/bin/env python3
"""
Data preparation script for PaddleOCR recognition finetuning.

Converts CSV ground truth files to PaddleOCR format (image_path\tlabel).
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data for PaddleOCR recognition finetuning"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--ground_truth_csv",
        type=str,
        required=True,
        help="CSV file with Filename,Label columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of using symlinks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    return parser.parse_args()


def read_csv_ground_truth(csv_path):
    """Read ground truth from CSV file with Filename,Label format."""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("Filename") or row.get("filename")
            label = row.get("Label") or row.get("label")
            if filename and label:
                data.append((filename, label))
    return data


def prepare_data(
    images_dir, ground_truth_csv, output_dir, train_ratio=0.8, copy_images=False, seed=42
):
    """
    Prepare data for PaddleOCR recognition training.

    Args:
        images_dir: Directory containing source images
        ground_truth_csv: CSV file with Filename,Label columns
        output_dir: Output directory for prepared data
        train_ratio: Ratio of data for training (rest for validation)
        copy_images: If True, copy images; otherwise create symlinks
        seed: Random seed for reproducible splits

    Returns:
        dict with paths to created files and statistics
    """
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Create output directories
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Read ground truth
    data = read_csv_ground_truth(ground_truth_csv)
    if not data:
        raise ValueError(f"No data found in {ground_truth_csv}")

    # Verify images exist and filter out missing ones
    valid_data = []
    missing_images = []
    for filename, label in data:
        img_path = images_dir / filename
        if img_path.exists():
            valid_data.append((filename, label))
        else:
            missing_images.append(filename)

    if missing_images:
        print(f"Warning: {len(missing_images)} images not found:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")

    if not valid_data:
        raise ValueError("No valid image-label pairs found")

    # Shuffle and split data
    random.seed(seed)
    random.shuffle(valid_data)

    split_idx = int(len(valid_data) * train_ratio)
    train_data = valid_data[:split_idx]
    val_data = valid_data[split_idx:]

    # Ensure at least 1 sample in each split
    if len(train_data) == 0 and len(valid_data) > 0:
        train_data = [valid_data[0]]
        val_data = valid_data[1:] if len(valid_data) > 1 else []
    if len(val_data) == 0 and len(valid_data) > 1:
        val_data = [train_data.pop()]

    # Copy/link images and create label files
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

    stats = {
        "total_samples": len(valid_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "missing_images": len(missing_images),
        "output_dir": str(output_dir),
        "train_list": str(train_list_path),
        "val_list": str(val_list_path),
    }

    return stats


def main():
    args = parse_args()

    print(f"Preparing data for PaddleOCR recognition finetuning...")
    print(f"  Images dir: {args.images_dir}")
    print(f"  Ground truth: {args.ground_truth_csv}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Train ratio: {args.train_ratio}")

    stats = prepare_data(
        images_dir=args.images_dir,
        ground_truth_csv=args.ground_truth_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        copy_images=args.copy_images,
        seed=args.seed,
    )

    print(f"\nData preparation complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Train samples: {stats['train_samples']}")
    print(f"  Val samples: {stats['val_samples']}")
    if stats["missing_images"] > 0:
        print(f"  Missing images: {stats['missing_images']}")
    print(f"\nOutput files:")
    print(f"  {stats['train_list']}")
    print(f"  {stats['val_list']}")


if __name__ == "__main__":
    main()
