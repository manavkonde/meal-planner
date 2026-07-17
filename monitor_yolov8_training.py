"""
Day 9 - Review YOLOv8 training progress and label the best checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUN_DIR = Path("models/checkpoints/yolov8n_fridge_v1")
DEFAULT_DATA = Path("data/processed/data.yaml")


@dataclass
class TrainingSummary:
    latest_epoch: int
    map50: float | None
    map5095: float | None
    train_box_delta: float | None
    train_cls_delta: float | None
    val_box_delta: float | None
    val_cls_delta: float | None
    decision: str


def clean_key(key: str) -> str:
    return key.strip()


def read_results(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [{clean_key(key): value for key, value in row.items()} for row in reader]


def get_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def metric_delta(rows: list[dict[str, str]], key: str) -> float | None:
    first = get_float(rows[0], key)
    last = get_float(rows[-1], key)
    if first is None or last is None:
        return None
    return last - first


def decide(summary_map50: float | None, latest_epoch: int, train_cls_delta: float | None) -> str:
    if latest_epoch < 5:
        return "Too early to decide; let nano continue until more epochs complete."
    if summary_map50 is None:
        return "Metrics unavailable; inspect training output before changing model size."
    if latest_epoch >= 30 and summary_map50 < 0.4 and (train_cls_delta is None or train_cls_delta >= 0):
        return "Escalate to yolov8s.pt; nano appears flat below the Day 9 threshold."
    if latest_epoch >= 30 and summary_map50 >= 0.5:
        return "Continue with yolov8n.pt; mAP@0.5 is trending toward the MVP target."
    return "Continue with yolov8n.pt for now; revisit the small model after more epochs."


def summarize(run_dir: Path) -> TrainingSummary:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results file: {results_csv}")

    rows = read_results(results_csv)
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")

    latest = rows[-1]
    latest_epoch = int(float(latest["epoch"]))
    map50 = get_float(latest, "metrics/mAP50(B)", "metrics/mAP50")
    map5095 = get_float(latest, "metrics/mAP50-95(B)", "metrics/mAP50-95")
    train_box_delta = metric_delta(rows, "train/box_loss")
    train_cls_delta = metric_delta(rows, "train/cls_loss")
    val_box_delta = metric_delta(rows, "val/box_loss")
    val_cls_delta = metric_delta(rows, "val/cls_loss")
    decision = decide(map50, latest_epoch, train_cls_delta)

    return TrainingSummary(
        latest_epoch=latest_epoch,
        map50=map50,
        map5095=map5095,
        train_box_delta=train_box_delta,
        train_cls_delta=train_cls_delta,
        val_box_delta=val_box_delta,
        val_cls_delta=val_cls_delta,
        decision=decision,
    )


def print_summary(summary: TrainingSummary, run_dir: Path) -> None:
    print(f"Run: {run_dir}")
    print(f"Latest epoch: {summary.latest_epoch}")
    print(f"mAP@0.5: {summary.map50 if summary.map50 is not None else 'n/a'}")
    print(f"mAP@0.5:0.95: {summary.map5095 if summary.map5095 is not None else 'n/a'}")
    print(f"train/box_loss delta: {summary.train_box_delta if summary.train_box_delta is not None else 'n/a'}")
    print(f"train/cls_loss delta: {summary.train_cls_delta if summary.train_cls_delta is not None else 'n/a'}")
    print(f"val/box_loss delta: {summary.val_box_delta if summary.val_box_delta is not None else 'n/a'}")
    print(f"val/cls_loss delta: {summary.val_cls_delta if summary.val_cls_delta is not None else 'n/a'}")
    print(f"Decision: {summary.decision}")


def copy_best_checkpoint(run_dir: Path, output: Path | None) -> Path:
    best_checkpoint = run_dir / "weights" / "best.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_checkpoint}")

    labeled_checkpoint = output or run_dir.with_name(f"{run_dir.name}_day9_best.pt")
    labeled_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, labeled_checkpoint)
    return labeled_checkpoint


def run_validation(weights: Path, data: Path) -> None:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(data))
    print(f"Interim val mAP@0.5: {metrics.box.map50}")
    print(f"Interim val mAP@0.5:0.95: {metrics.box.map}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review Day 9 YOLOv8 training progress.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--copy-best", action="store_true")
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--val", action="store_true", help="Run YOLO validation using weights/best.pt.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(args.run_dir)
    print_summary(summary, args.run_dir)

    best_checkpoint = args.run_dir / "weights" / "best.pt"
    last_checkpoint = args.run_dir / "weights" / "last.pt"
    print(f"best.pt exists: {best_checkpoint.exists()}")
    print(f"last.pt exists: {last_checkpoint.exists()}")

    if args.copy_best:
        copied_to = copy_best_checkpoint(args.run_dir, args.best_output)
        print(f"Copied best checkpoint to: {copied_to}")

    if args.val:
        run_validation(best_checkpoint, args.data)


if __name__ == "__main__":
    main()
