#!/usr/bin/env python3
"""
Convert PaddleOCR training logs to TensorBoard format.

This script parses train.log files from PaddleOCR training runs and converts
them to TensorBoard event files for visualization.

Usage:
    python tools/log_to_tensorboard.py \
        --log ./output/server_rec_full/models/train.log \
        --output ./tensorboard_logs/server_rec_full

    # Then view with TensorBoard:
    tensorboard --logdir ./tensorboard_logs
"""

import argparse
import re
from pathlib import Path
from datetime import datetime


def parse_training_step(line: str) -> dict | None:
    """
    Parse a training step log line.

    Example line:
    [2026/01/30 01:23:26] ppocr INFO: epoch: [1/200], global_step: 5, lr: 0.000000,
    acc: 0.000000, norm_edit_dis: 0.220916, CTCLoss: 71.444366, NRTRLoss: 4.117732,
    loss: 75.562096, avg_reader_cost: 0.14362 s, avg_batch_cost: 0.82421 s,
    avg_samples: 64.0, ips: 77.65025 samples/s, eta: 6:02:34,
    max_mem_reserved: 2029 MB, max_mem_allocated: 1746 MB
    """
    if "global_step:" not in line:
        return None

    result = {}

    # Parse timestamp
    timestamp_match = re.match(r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", line)
    if timestamp_match:
        result["timestamp"] = datetime.strptime(
            timestamp_match.group(1), "%Y/%m/%d %H:%M:%S"
        )

    # Parse epoch
    epoch_match = re.search(r"epoch: \[(\d+)/(\d+)\]", line)
    if epoch_match:
        result["epoch"] = int(epoch_match.group(1))
        result["total_epochs"] = int(epoch_match.group(2))

    # Parse global_step
    step_match = re.search(r"global_step: (\d+)", line)
    if step_match:
        result["global_step"] = int(step_match.group(1))

    # Parse numeric values with regex
    patterns = {
        "lr": r"lr: ([\d.e-]+)",
        "acc": r"(?<!norm_edit_dis: .{0,20})acc: ([\d.]+)",  # Avoid matching inside norm_edit_dis
        "norm_edit_dis": r"norm_edit_dis: ([\d.]+)",
        "CTCLoss": r"CTCLoss: ([\d.]+)",
        "NRTRLoss": r"NRTRLoss: ([\d.]+)",
        "loss": r"(?<![A-Z])loss: ([\d.]+)",  # Total loss, not CTCLoss/NRTRLoss
        "ips": r"ips: ([\d.]+)",
        "max_mem_reserved": r"max_mem_reserved: (\d+)",
        "max_mem_allocated": r"max_mem_allocated: (\d+)",
    }

    # Simplified approach - extract key-value pairs
    # lr
    m = re.search(r", lr: ([\d.e-]+),", line)
    if m:
        result["lr"] = float(m.group(1))

    # acc (training accuracy, not from norm_edit_dis context)
    m = re.search(r", acc: ([\d.]+),", line)
    if m:
        result["acc"] = float(m.group(1))

    # norm_edit_dis
    m = re.search(r"norm_edit_dis: ([\d.]+),", line)
    if m:
        result["norm_edit_dis"] = float(m.group(1))

    # CTCLoss
    m = re.search(r"CTCLoss: ([\d.]+)", line)
    if m:
        result["CTCLoss"] = float(m.group(1))

    # NRTRLoss
    m = re.search(r"NRTRLoss: ([\d.]+)", line)
    if m:
        result["NRTRLoss"] = float(m.group(1))

    # Total loss (comes after NRTRLoss)
    m = re.search(r"NRTRLoss: [\d.]+, loss: ([\d.]+)", line)
    if m:
        result["loss"] = float(m.group(1))

    # ips (images per second)
    m = re.search(r"ips: ([\d.]+) samples/s", line)
    if m:
        result["ips"] = float(m.group(1))

    # Memory usage
    m = re.search(r"max_mem_reserved: (\d+) MB", line)
    if m:
        result["max_mem_reserved_MB"] = int(m.group(1))

    m = re.search(r"max_mem_allocated: (\d+) MB", line)
    if m:
        result["max_mem_allocated_MB"] = int(m.group(1))

    # Only return if we got essential fields
    if "global_step" in result:
        return result
    return None


def parse_eval_metric(line: str) -> dict | None:
    """
    Parse a validation metric log line.

    Example line:
    [2026/01/30 01:28:09] ppocr INFO: cur metric, acc: 0.7223007981032493,
    norm_edit_dis: 0.9605192834069811, fps: 709.6428926176906
    """
    if "cur metric, acc:" not in line:
        return None

    result = {}

    # Parse timestamp
    timestamp_match = re.match(r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", line)
    if timestamp_match:
        result["timestamp"] = datetime.strptime(
            timestamp_match.group(1), "%Y/%m/%d %H:%M:%S"
        )

    # Parse accuracy
    m = re.search(r"cur metric, acc: ([\d.]+)", line)
    if m:
        result["eval_acc"] = float(m.group(1))

    # Parse norm_edit_dis
    m = re.search(r"norm_edit_dis: ([\d.]+)", line)
    if m:
        result["eval_norm_edit_dis"] = float(m.group(1))

    # Parse fps
    m = re.search(r"fps: ([\d.]+)", line)
    if m:
        result["eval_fps"] = float(m.group(1))

    if "eval_acc" in result:
        return result
    return None


def parse_best_metric(line: str) -> dict | None:
    """
    Parse a best metric log line.

    Example line:
    [2026/01/30 01:28:17] ppocr INFO: best metric, acc: 0.7223007981032493,
    is_float16: False, norm_edit_dis: 0.9605192834069811,
    fps: 709.6428926176906, best_epoch: 4
    """
    if "best metric, acc:" not in line:
        return None

    result = {}

    # Parse timestamp
    timestamp_match = re.match(r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", line)
    if timestamp_match:
        result["timestamp"] = datetime.strptime(
            timestamp_match.group(1), "%Y/%m/%d %H:%M:%S"
        )

    # Parse best accuracy
    m = re.search(r"best metric, acc: ([\d.]+)", line)
    if m:
        result["best_acc"] = float(m.group(1))

    # Parse norm_edit_dis
    m = re.search(r"norm_edit_dis: ([\d.]+)", line)
    if m:
        result["best_norm_edit_dis"] = float(m.group(1))

    # Parse best_epoch
    m = re.search(r"best_epoch: (\d+)", line)
    if m:
        result["best_epoch"] = int(m.group(1))

    if "best_acc" in result:
        return result
    return None


def parse_log_file(log_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse a PaddleOCR training log file.

    Returns:
        Tuple of (training_steps, eval_metrics, best_metrics)
    """
    training_steps = []
    eval_metrics = []
    best_metrics = []

    with open(log_path, "r") as f:
        for line in f:
            # Try parsing as training step
            train_data = parse_training_step(line)
            if train_data:
                training_steps.append(train_data)
                continue

            # Try parsing as eval metric
            eval_data = parse_eval_metric(line)
            if eval_data:
                eval_metrics.append(eval_data)
                continue

            # Try parsing as best metric
            best_data = parse_best_metric(line)
            if best_data:
                best_metrics.append(best_data)

    return training_steps, eval_metrics, best_metrics


def write_tensorboard_logs(
    output_dir: Path,
    training_steps: list[dict],
    eval_metrics: list[dict],
    best_metrics: list[dict],
) -> None:
    """
    Write parsed metrics to TensorBoard event files.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                "Please install tensorboard: pip install tensorboard\n"
                "Or tensorboardX: pip install tensorboardX"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir))

    # Write training metrics
    print(f"Writing {len(training_steps)} training steps...")
    for step_data in training_steps:
        global_step = step_data.get("global_step", 0)

        if "loss" in step_data:
            writer.add_scalar("Train/loss", step_data["loss"], global_step)
        if "CTCLoss" in step_data:
            writer.add_scalar("Train/CTCLoss", step_data["CTCLoss"], global_step)
        if "NRTRLoss" in step_data:
            writer.add_scalar("Train/NRTRLoss", step_data["NRTRLoss"], global_step)
        if "acc" in step_data:
            writer.add_scalar("Train/acc", step_data["acc"], global_step)
        if "norm_edit_dis" in step_data:
            writer.add_scalar("Train/norm_edit_dis", step_data["norm_edit_dis"], global_step)
        if "lr" in step_data:
            writer.add_scalar("Train/lr", step_data["lr"], global_step)
        if "ips" in step_data:
            writer.add_scalar("Train/samples_per_sec", step_data["ips"], global_step)
        if "max_mem_reserved_MB" in step_data:
            writer.add_scalar("Train/memory_reserved_MB", step_data["max_mem_reserved_MB"], global_step)
        if "max_mem_allocated_MB" in step_data:
            writer.add_scalar("Train/memory_allocated_MB", step_data["max_mem_allocated_MB"], global_step)

    # Write eval metrics (indexed by evaluation number since we don't have step info)
    print(f"Writing {len(eval_metrics)} evaluation metrics...")
    for i, eval_data in enumerate(eval_metrics):
        if "eval_acc" in eval_data:
            writer.add_scalar("Eval/acc", eval_data["eval_acc"], i)
        if "eval_norm_edit_dis" in eval_data:
            writer.add_scalar("Eval/norm_edit_dis", eval_data["eval_norm_edit_dis"], i)
        if "eval_fps" in eval_data:
            writer.add_scalar("Eval/fps", eval_data["eval_fps"], i)

    # Write best metrics
    print(f"Writing {len(best_metrics)} best metric updates...")
    for i, best_data in enumerate(best_metrics):
        if "best_acc" in best_data:
            writer.add_scalar("Best/acc", best_data["best_acc"], i)
        if "best_norm_edit_dis" in best_data:
            writer.add_scalar("Best/norm_edit_dis", best_data["best_norm_edit_dis"], i)
        if "best_epoch" in best_data:
            writer.add_scalar("Best/epoch", best_data["best_epoch"], i)

    writer.close()
    print(f"TensorBoard logs written to: {output_dir}")


def print_summary(
    training_steps: list[dict],
    eval_metrics: list[dict],
    best_metrics: list[dict],
) -> None:
    """Print a summary of the parsed metrics."""
    print("\n" + "=" * 60)
    print("TRAINING LOG SUMMARY")
    print("=" * 60)

    print(f"\nTotal training steps logged: {len(training_steps)}")
    print(f"Total evaluation runs: {len(eval_metrics)}")
    print(f"Best metric updates: {len(best_metrics)}")

    if training_steps:
        first_step = training_steps[0]
        last_step = training_steps[-1]
        print(f"\nTraining range:")
        print(f"  First step: {first_step.get('global_step', 'N/A')}")
        print(f"  Last step: {last_step.get('global_step', 'N/A')}")
        if "epoch" in first_step and "epoch" in last_step:
            print(f"  Epochs: {first_step['epoch']} -> {last_step['epoch']}")

        # Loss progression
        if "loss" in first_step and "loss" in last_step:
            print(f"\nLoss progression:")
            print(f"  Start: {first_step['loss']:.4f}")
            print(f"  End: {last_step['loss']:.4f}")

    if best_metrics:
        # Find unique best accuracy values to show progression
        unique_bests = []
        last_acc = None
        for bm in best_metrics:
            acc = bm.get("best_acc")
            if acc and acc != last_acc:
                unique_bests.append(bm)
                last_acc = acc

        print(f"\nBest accuracy progression ({len(unique_bests)} improvements):")
        for bm in unique_bests:
            epoch = bm.get("best_epoch", "?")
            acc = bm.get("best_acc", 0)
            print(f"  Epoch {epoch:3d}: {acc*100:.2f}%")

        if unique_bests:
            final_best = unique_bests[-1]
            print(f"\nFinal best accuracy: {final_best.get('best_acc', 0)*100:.2f}% (epoch {final_best.get('best_epoch', '?')})")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR training logs to TensorBoard format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a training log
  python tools/log_to_tensorboard.py \\
      --log ./output/server_rec_full/models/train.log \\
      --output ./tensorboard_logs/server_rec_full

  # View with TensorBoard
  tensorboard --logdir ./tensorboard_logs

  # Just print summary without creating TensorBoard logs
  python tools/log_to_tensorboard.py \\
      --log ./output/server_rec_full/models/train.log \\
      --summary-only
        """
    )

    parser.add_argument(
        "--log", "-l",
        type=Path,
        required=True,
        help="Path to PaddleOCR train.log file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for TensorBoard event files"
    )
    parser.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Only print summary, don't create TensorBoard logs"
    )

    args = parser.parse_args()

    if not args.log.exists():
        print(f"Error: Log file not found: {args.log}")
        return 1

    print(f"Parsing log file: {args.log}")
    training_steps, eval_metrics, best_metrics = parse_log_file(args.log)

    # Print summary
    print_summary(training_steps, eval_metrics, best_metrics)

    if not args.summary_only:
        if args.output is None:
            # Default output path
            args.output = Path("./tensorboard_logs") / args.log.parent.name

        write_tensorboard_logs(args.output, training_steps, eval_metrics, best_metrics)
        print(f"\nTo view, run: tensorboard --logdir {args.output.parent}")

    return 0


if __name__ == "__main__":
    exit(main())
