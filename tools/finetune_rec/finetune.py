#!/usr/bin/env python3
"""
Main finetuning script for PP-OCRv5 recognition models (mobile or server).

This script:
1. Prepares data from CSV ground truth (single CSV auto-split or pre-split CSVs)
2. Downloads pretrained model if needed
3. Runs training with the finetuning config

Supports two model variants:
- --model mobile: Lighter, faster model (PPLCNetV3 backbone)
- --model server: Larger, more accurate model (PPHGNetV2_B4 backbone)

Supports two data input modes:
- Single CSV with auto-split: --ground_truth_csv (uses --train_ratio for split)
- Pre-split CSVs: --train_csv and --val_csv

Supports preset configs:
- --preset quick: 2 epochs for quick pipeline testing
- --preset full: 200 epochs for full training
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PADDLEOCR_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PADDLEOCR_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_data import prepare_data
from prepare_data_presplit import prepare_data_presplit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune PP-OCRv5 recognition model (mobile or server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single CSV with auto-split (quick test, mobile model)
  python finetune.py --images_dir ./images --ground_truth_csv ./labels.csv \\
      --output_dir ./output/test --preset quick

  # Pre-split CSVs (full training, mobile model)
  python finetune.py --images_dir ./images --train_csv ./train.csv \\
      --val_csv ./val.csv --output_dir ./output/full --preset full

  # Server model (larger, more accurate)
  python finetune.py --images_dir ./images --train_csv ./train.csv \\
      --val_csv ./val.csv --output_dir ./output/full --preset full --model server

  # Resume training
  python finetune.py --images_dir ./images --train_csv ./train.csv \\
      --val_csv ./val.csv --output_dir ./output/full --preset full \\
      --resume ./output/full/models/latest
""",
    )

    # Data input arguments (mutually exclusive modes)
    data_group = parser.add_argument_group("Data Input (choose one mode)")
    data_group.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    data_group.add_argument(
        "--ground_truth_csv",
        type=str,
        default=None,
        help="Single CSV file with Filename,Label columns (auto-split mode)",
    )
    data_group.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Training CSV file (pre-split mode, requires --val_csv)",
    )
    data_group.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Validation CSV file (pre-split mode, requires --train_csv)",
    )
    data_group.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training in auto-split mode (default: 0.8)",
    )

    # Output and model arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model and prepared data",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mobile", "server"],
        default="mobile",
        help="Model variant: 'mobile' (faster) or 'server' (more accurate). Default: mobile",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model (will download if not specified)",
    )

    # Config arguments
    config_group = parser.add_argument_group("Config")
    config_group.add_argument(
        "--preset",
        type=str,
        choices=["quick", "full"],
        default=None,
        help="Use preset config: 'quick' (2 epochs) or 'full' (200 epochs)",
    )
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Custom config file (overrides --preset)",
    )

    # Training overrides
    train_group = parser.add_argument_group("Training Overrides")
    train_group.add_argument(
        "--epoch_num",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per card (overrides config)",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    train_group.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use, e.g., '0' or '0,1,2,3' (default: '0')",
    )
    train_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    train_group.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, no training",
    )

    args = parser.parse_args()

    # Validate mutually exclusive data input modes
    has_single_csv = args.ground_truth_csv is not None
    has_presplit = args.train_csv is not None or args.val_csv is not None

    if has_single_csv and has_presplit:
        parser.error(
            "--ground_truth_csv cannot be used with --train_csv/--val_csv. "
            "Choose one mode: single CSV (auto-split) or pre-split CSVs."
        )

    if not has_single_csv and not has_presplit:
        parser.error(
            "Must provide either --ground_truth_csv (auto-split mode) or "
            "--train_csv and --val_csv (pre-split mode)."
        )

    if has_presplit:
        if args.train_csv is None:
            parser.error("--train_csv is required when using pre-split mode.")
        if args.val_csv is None:
            parser.error("--val_csv is required when using --train_csv.")

    return args


def download_pretrained_model(output_dir, model_type="mobile"):
    """Download pretrained PP-OCRv5 rec model using PaddleOCR API."""
    model_name = f"PP-OCRv5_{model_type}_rec"
    print(f"Downloading pretrained {model_name} model...")

    # Use paddleocr to trigger model download
    try:
        from paddleocr import TextRecognition

        # This will download the model to ~/.paddlex/official_models/
        rec = TextRecognition(model_name=model_name)

        # Find the downloaded model path
        paddlex_model_dir = Path.home() / ".paddlex" / "official_models" / model_name
        if paddlex_model_dir.exists():
            # Look for the inference model or best_accuracy
            for subdir in ["best_accuracy", "inference", ""]:
                model_path = paddlex_model_dir / subdir if subdir else paddlex_model_dir
                if (model_path / "inference.pdmodel").exists():
                    print(f"Found inference model at: {model_path}")
                    return str(model_path)
                if (model_path / "model.pdparams").exists():
                    print(f"Found training model at: {model_path}")
                    return str(model_path)

        # Alternative: check for model in paddleocr cache
        print(f"Model downloaded. Using path: {paddlex_model_dir}")
        return str(paddlex_model_dir)

    except Exception as e:
        print(f"Warning: Could not download model via PaddleOCR API: {e}")
        print("Please download the model manually and specify --pretrained_model")
        return None


def find_pretrained_model(model_type="mobile"):
    """Find pretrained model in common locations.

    Returns the model path WITHOUT the .pdparams extension,
    as PaddleOCR's load_model adds the extension automatically.
    """
    model_name = f"PP-OCRv5_{model_type}_rec"
    common_paths = [
        Path.home() / ".paddlex" / "official_models" / model_name,
        Path(f"./pretrain_models/{model_name}"),
        Path(f"./pretrain_models/{model_name}/best_accuracy"),
    ]

    for path in common_paths:
        if path.exists():
            # Check for model.pdparams - return path/model (without extension)
            if (path / "model.pdparams").exists():
                return str(path / "model")
            # Check for best_accuracy.pdparams
            if (path / "best_accuracy.pdparams").exists():
                return str(path / "best_accuracy")
            # Check subdirectories
            for subdir in ["best_accuracy", "inference"]:
                subpath = path / subdir
                if subpath.exists():
                    if (subpath / "model.pdparams").exists():
                        return str(subpath / "model")
                    if (subpath / "best_accuracy.pdparams").exists():
                        return str(subpath / "best_accuracy")

    return None


def resolve_config_path(args):
    """Resolve config path based on --config, --preset, and --model arguments."""
    if args.config:
        # Custom config takes precedence
        return Path(args.config)

    config_dir = PADDLEOCR_ROOT / "configs" / "rec" / "PP-OCRv5"
    model_type = args.model  # "mobile" or "server"

    if args.preset == "quick":
        return config_dir / f"PP-OCRv5_{model_type}_rec_finetune_quick.yml"
    elif args.preset == "full":
        return config_dir / f"PP-OCRv5_{model_type}_rec_finetune_full.yml"
    else:
        # Default config (full for server, base for mobile)
        if model_type == "server":
            return config_dir / "PP-OCRv5_server_rec_finetune_full.yml"
        return config_dir / "PP-OCRv5_mobile_rec_finetune.yml"


def run_training(
    config_path,
    data_dir,
    output_dir,
    pretrained_model=None,
    epoch_num=None,
    batch_size=None,
    learning_rate=None,
    gpus="0",
    checkpoint=None,
    eval_only=False,
):
    """Run PaddleOCR training."""
    # Build command
    train_script = PADDLEOCR_ROOT / "tools" / ("eval.py" if eval_only else "train.py")

    # Check if multi-GPU
    gpu_list = gpus.split(",")
    python_exe = sys.executable
    if len(gpu_list) > 1:
        cmd = [
            python_exe,
            "-m",
            "paddle.distributed.launch",
            f"--gpus={gpus}",
            str(train_script),
        ]
    else:
        cmd = [python_exe, str(train_script)]

    cmd.extend(["-c", str(config_path)])

    # Add overrides
    overrides = []

    # Data paths
    overrides.append(f"Train.dataset.data_dir={data_dir}")
    overrides.append(f"Train.dataset.label_file_list=['{data_dir}/train_list.txt']")
    overrides.append(f"Eval.dataset.data_dir={data_dir}")
    overrides.append(f"Eval.dataset.label_file_list=['{data_dir}/val_list.txt']")
    overrides.append(f"Global.save_model_dir={output_dir}/models")

    if pretrained_model and not checkpoint:
        overrides.append(f"Global.pretrained_model={pretrained_model}")

    if checkpoint:
        overrides.append(f"Global.checkpoints={checkpoint}")

    if epoch_num is not None:
        overrides.append(f"Global.epoch_num={epoch_num}")

    if batch_size is not None:
        overrides.append(f"Train.loader.batch_size_per_card={batch_size}")
        overrides.append(f"Eval.loader.batch_size_per_card={batch_size}")

    if learning_rate is not None:
        overrides.append(f"Optimizer.lr.learning_rate={learning_rate}")

    # Single GPU mode
    if len(gpu_list) == 1:
        overrides.append("Global.distributed=false")

    if overrides:
        cmd.append("-o")
        cmd.extend(overrides)

    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    # Run training
    result = subprocess.run(cmd, env=env, cwd=str(PADDLEOCR_ROOT))
    return result.returncode


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    data_dir = output_dir / "data"

    model_type = args.model
    model_label = "Server" if model_type == "server" else "Mobile"

    print("=" * 60)
    print(f"PP-OCRv5 {model_label} Recognition Finetuning")
    print("=" * 60)

    # Determine data mode
    is_presplit_mode = args.train_csv is not None

    if is_presplit_mode:
        print(f"Mode: Pre-split CSVs")
        print(f"  Train CSV: {args.train_csv}")
        print(f"  Val CSV: {args.val_csv}")
    else:
        print(f"Mode: Single CSV (auto-split)")
        print(f"  Ground truth: {args.ground_truth_csv}")
        print(f"  Train ratio: {args.train_ratio}")

    # Show preset info
    if args.preset:
        print(f"Preset: {args.preset}")
    elif args.config:
        print(f"Config: {args.config}")

    # Step 1: Prepare data
    print("\n[Step 1/3] Preparing data...")

    if is_presplit_mode:
        # Pre-split mode
        stats = prepare_data_presplit(
            images_dir=args.images_dir,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            output_dir=str(data_dir),
        )
        train_samples = stats["final_train_count"]
        val_samples = stats["final_val_count"]

        # Print cleaning summary
        print(f"\n  Cleaning summary:")
        if stats["train_duplicates_removed"] > 0:
            print(f"    Train duplicates removed: {stats['train_duplicates_removed']}")
        if stats["val_duplicates_removed"] > 0:
            print(f"    Val duplicates removed: {stats['val_duplicates_removed']}")
        if stats["cross_contamination_count"] > 0:
            print(f"    Cross-contamination fixed: {stats['cross_contamination_count']}")
        if stats["train_missing_count"] > 0:
            print(f"    Train missing images: {stats['train_missing_count']}")
        if stats["val_missing_count"] > 0:
            print(f"    Val missing images: {stats['val_missing_count']}")
        print(f"  Report: {stats['report_path']}")
    else:
        # Auto-split mode
        stats = prepare_data(
            images_dir=args.images_dir,
            ground_truth_csv=args.ground_truth_csv,
            output_dir=str(data_dir),
            train_ratio=args.train_ratio,
        )
        train_samples = stats["train_samples"]
        val_samples = stats["val_samples"]

    print(f"\n  Final counts:")
    print(f"    Train samples: {train_samples}")
    print(f"    Val samples: {val_samples}")

    if train_samples < 10:
        print(
            f"\n  Warning: Very small training set ({train_samples} samples)."
        )
        print("  Results may be poor. PaddleOCR recommends 5000+ samples for finetuning.")

    # Step 2: Get pretrained model
    print("\n[Step 2/3] Setting up pretrained model...")
    pretrained_model = args.pretrained_model

    if not pretrained_model:
        pretrained_model = find_pretrained_model(model_type)

    if not pretrained_model:
        print("Pretrained model not found locally. Downloading...")
        pretrained_model = download_pretrained_model(output_dir, model_type)

    if pretrained_model:
        print(f"  Using pretrained model: {pretrained_model}")
    else:
        print("  Warning: No pretrained model found. Training from scratch.")

    # Step 3: Run training
    print("\n[Step 3/3] Running training...")

    # Resolve config path
    config_path = resolve_config_path(args)
    print(f"  Config: {config_path}")

    return_code = run_training(
        config_path=config_path,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        pretrained_model=pretrained_model,
        epoch_num=args.epoch_num,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gpus=args.gpus,
        checkpoint=args.resume,
        eval_only=args.eval_only,
    )

    if return_code == 0:
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Models saved to: {output_dir}/models")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"Training failed with return code: {return_code}")
        print("=" * 60)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
