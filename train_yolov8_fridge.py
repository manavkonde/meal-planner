"""
Day 8 - YOLOv8 fine-tuning launcher for the fridge/pantry dataset.

Runs data readiness checks, reports CUDA availability, and optionally launches
the first transfer-learning run from COCO-pretrained YOLOv8 weights.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA = Path("data/processed/data.yaml")
DEFAULT_WEIGHTS = Path("yolov8n.pt")
DEFAULT_PROJECT = Path("models/checkpoints")
DEFAULT_NAME = "yolov8n_fridge_v1"


@dataclass
class DatasetConfig:
    root: Path
    train: Path
    val: Path
    test: Path | None
    names: list[str]
    nc: int


def parse_data_yaml(data_yaml: Path) -> DatasetConfig:
    values: dict[str, str] = {}

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    missing = {"path", "train", "val", "nc", "names"} - values.keys()
    if missing:
        raise ValueError(f"{data_yaml} is missing required keys: {sorted(missing)}")

    root = Path(values["path"]).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    names = ast.literal_eval(values["names"])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("data.yaml names must be a list of class-name strings")

    nc = int(values["nc"])
    if nc != len(names):
        raise ValueError(f"data.yaml nc={nc} but names contains {len(names)} classes")

    def split_path(split_name: str) -> Path | None:
        split_value = values.get(split_name)
        if not split_value:
            return None
        split = Path(split_value)
        return split if split.is_absolute() else root / split

    return DatasetConfig(
        root=root,
        train=split_path("train") or root / "images/train",
        val=split_path("val") or root / "images/val",
        test=split_path("test"),
        names=names,
        nc=nc,
    )


def count_files(directory: Path, extensions: set[str] | None = None) -> int:
    if not directory.exists():
        return 0
    files = (path for path in directory.iterdir() if path.is_file())
    if extensions is None:
        return sum(1 for _ in files)
    return sum(1 for path in files if path.suffix.lower() in extensions)


def validate_labels(labels_dir: Path, nc: int) -> tuple[int, int]:
    checked_files = 0
    checked_boxes = 0

    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    for label_path in labels_dir.glob("*.txt"):
        checked_files += 1
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{label_path}:{line_number} has fewer than 5 YOLO fields")
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:5]]
            if class_id < 0 or class_id >= nc:
                raise ValueError(f"{label_path}:{line_number} class id {class_id} outside 0..{nc - 1}")
            if any(value < 0 or value > 1 for value in coords):
                raise ValueError(f"{label_path}:{line_number} has normalized coordinates outside 0..1")
            checked_boxes += 1

    return checked_files, checked_boxes


def check_dataset(config: DatasetConfig) -> None:
    print(f"Dataset root: {config.root}")
    print(f"Classes: {config.nc}")

    for split_name, images_dir in (("train", config.train), ("val", config.val), ("test", config.test)):
        if images_dir is None:
            continue
        labels_dir = config.root / "labels" / split_name
        image_count = count_files(images_dir, IMAGE_EXTENSIONS)
        label_count = count_files(labels_dir, {".txt"})
        print(f"{split_name}: {image_count} images, {label_count} labels")
        if image_count == 0:
            raise FileNotFoundError(f"No images found for {split_name}: {images_dir}")
        if label_count == 0:
            raise FileNotFoundError(f"No labels found for {split_name}: {labels_dir}")
        checked_files, checked_boxes = validate_labels(labels_dir, config.nc)
        print(f"{split_name}: validated {checked_files} label files, {checked_boxes} boxes")


def report_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("PyTorch not installed; install requirements before training.")
        return False

    available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if available else "No GPU"
    print(f"CUDA available: {available} ({device_name})")
    return available


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Day 8 YOLOv8 fridge fine-tuning.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run checks without starting training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = parse_data_yaml(args.data)
    check_dataset(config)
    has_cuda = report_cuda()

    if args.dry_run:
        print("Dry run complete. Training was not started.")
        return

    if not args.weights.exists() and not str(args.weights).startswith("yolo"):
        raise FileNotFoundError(f"Missing starting weights: {args.weights}")

    if not has_cuda:
        print("No local CUDA GPU detected. Training can run on CPU, but Colab/T4 is recommended.")

    print(
        "Starting training with "
        f"weights={args.weights}, epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, patience={args.patience}"
    )
    train(args)


if __name__ == "__main__":
    main()
