"""
Day 3 — Task 1 & 2: Dataset Audit
Walks every dataset, records image metadata (path, label, width, height, valid),
saves per-dataset audit CSVs to data/processed/, and prints summary statistics.
"""

import os
import sys
import random
from PIL import Image
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Dataset configs: name → (path, label_extraction_function)
# Each label function takes (root, filename) and returns a label string.

def label_from_parent_dir(root, filename):
    """Label = parent directory name (used by fruits360)."""
    return os.path.basename(root)

def label_from_filename_prefix(root, filename):
    """Label = filename prefix before underscore (used by fridge_objects flat dir)."""
    # e.g. 'Banana_42.jpg' → 'Banana'
    parts = filename.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else os.path.basename(root)

def label_from_grocery_path(root, filename):
    """Label from grocery dataset path structure: .../train/Fruit/Apple/... → 'Apple'
    or from flat fridge dataset dir: filename prefix."""
    # The grocery dataset has structure: dataset/{train,test,val}/{Category}/{SubCategory}/
    # But fridge_objects has flat files like 'Banana_42.jpg'
    rel = os.path.relpath(root, RAW_DIR)
    parts = rel.replace("\\", "/").split("/")
    # grocery: grocery/dataset/{split}/{Category}/{SubCategory} → SubCategory
    # If the immediate parent is one of [train, test, val], use filename prefix
    parent = os.path.basename(root)
    if parent in ("train", "test", "val"):
        return label_from_filename_prefix(root, filename)
    return parent


DATASETS = {}

# ── Fruits-360 ────────────────────────────────────────────────────────────────
fruits_train = os.path.join(RAW_DIR, "fruits360", "Training")
fruits_test = os.path.join(RAW_DIR, "fruits360", "Test")
if os.path.isdir(fruits_train):
    DATASETS["fruits360"] = {
        "paths": [fruits_train, fruits_test] if os.path.isdir(fruits_test) else [fruits_train],
        "label_fn": label_from_parent_dir,
    }

# ── Grocery Store Dataset ────────────────────────────────────────────────────
grocery_ds = os.path.join(RAW_DIR, "grocery", "dataset")
if os.path.isdir(grocery_ds):
    grocery_paths = []
    for split in ("train", "test", "val"):
        sp = os.path.join(grocery_ds, split)
        if os.path.isdir(sp):
            grocery_paths.append(sp)
    if grocery_paths:
        DATASETS["grocery"] = {
            "paths": grocery_paths,
            "label_fn": label_from_grocery_path,
        }

# ── Fridge Object Detection ──────────────────────────────────────────────────
fridge_ds = os.path.join(RAW_DIR, "fridge_objects", "Refrigerator_Contents_7_Classes_Dataset",
                         "Refrigerator_Contents_7_Classes_Dataset")
if os.path.isdir(fridge_ds):
    DATASETS["fridge_objects"] = {
        "paths": [fridge_ds],
        "label_fn": label_from_filename_prefix,
    }


# ── Audit Function ───────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def audit_dataset(name, paths, label_fn):
    """Walk dataset folder(s), record image metadata."""
    print(f"\n{'='*60}")
    print(f"Auditing dataset: {name}")
    print(f"{'='*60}")

    stats = []
    for folder in paths:
        for root, dirs, files in os.walk(folder):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                path = os.path.join(root, f)
                try:
                    img = Image.open(path)
                    img.verify()  # Verify the image is not corrupted
                    # Re-open after verify (verify closes the file)
                    img = Image.open(path)
                    stats.append({
                        "path": path,
                        "label": label_fn(root, f),
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode,
                        "valid": True,
                    })
                except Exception as e:
                    stats.append({
                        "path": path,
                        "label": label_fn(root, f),
                        "width": None,
                        "height": None,
                        "format": None,
                        "mode": None,
                        "valid": False,
                        "error": str(e),
                    })

    df = pd.DataFrame(stats)
    if df.empty:
        print(f"  WARNING: No images found for {name}!")
        return df

    valid_count = df["valid"].sum()
    invalid_count = (~df["valid"]).sum()
    total = len(df)

    print(f"  Total images: {total}")
    print(f"  Valid images: {valid_count}")
    print(f"  Corrupted/unreadable: {invalid_count}")

    if invalid_count > 0:
        print(f"\n  Corrupted files:")
        for _, row in df[~df["valid"]].iterrows():
            print(f"    - {row['path']}: {row.get('error', 'unknown error')}")

    # Print unique labels count
    unique_labels = df[df["valid"]]["label"].nunique()
    print(f"  Unique labels/classes: {unique_labels}")

    # Print image dimension stats for valid images
    valid_df = df[df["valid"]]
    if not valid_df.empty:
        print(f"\n  Image dimensions (valid images):")
        print(f"    Width  — min: {valid_df['width'].min()}, max: {valid_df['width'].max()}, "
              f"mean: {valid_df['width'].mean():.0f}")
        print(f"    Height — min: {valid_df['height'].min()}, max: {valid_df['height'].max()}, "
              f"mean: {valid_df['height'].mean():.0f}")

    # Task 1: Display random sample info (10 images)
    sample_size = min(10, len(valid_df))
    if sample_size > 0:
        print(f"\n  Random sample of {sample_size} images:")
        sample = valid_df.sample(n=sample_size, random_state=42)
        for _, row in sample.iterrows():
            print(f"    [{row['label']}] {os.path.basename(row['path'])} "
                  f"— {row['width']}x{row['height']} {row['format']} {row['mode']}")

    # Save audit CSV
    csv_path = os.path.join(PROCESSED_DIR, f"audit_{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Audit CSV saved to: {csv_path}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    for name, config in DATASETS.items():
        df = audit_dataset(name, config["paths"], config["label_fn"])
        all_results[name] = df

    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    for name, df in all_results.items():
        if df.empty:
            print(f"  {name}: NO IMAGES FOUND")
        else:
            print(f"  {name}: {len(df)} total, {df['valid'].sum()} valid, "
                  f"{(~df['valid']).sum()} corrupted, "
                  f"{df[df['valid']]['label'].nunique()} classes")

    print("\nAll audit CSVs saved to data/processed/")
    print("Done!")
