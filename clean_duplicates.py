"""
Day 3 — Task 5: Remove Corrupted and Duplicate Images
- Quarantines corrupted files (valid==False) to a quarantine folder.
- Detects near-duplicate images using perceptual hashing (imagehash).
- Logs all removals.

NOTE: Per PRD guidance, raw data is kept untouched — corrupted files are MOVED
to a quarantine folder, not deleted.
"""

import os
import shutil
import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
QUARANTINE_DIR = os.path.join(BASE_DIR, "data", "quarantine")
os.makedirs(QUARANTINE_DIR, exist_ok=True)

DATASETS = ["fruits360", "grocery", "fridge_objects"]

# Hash distance threshold: images with hash distance <= this are considered duplicates
HASH_DISTANCE_THRESHOLD = 5


def quarantine_corrupted(name, df):
    """Move corrupted files to quarantine folder."""
    corrupted = df[df["valid"] == False]
    if corrupted.empty:
        print(f"  {name}: No corrupted files to quarantine.")
        return 0

    quarantine_ds = os.path.join(QUARANTINE_DIR, name)
    os.makedirs(quarantine_ds, exist_ok=True)

    moved = 0
    for _, row in corrupted.iterrows():
        src = row["path"]
        if os.path.exists(src):
            dst = os.path.join(quarantine_ds, os.path.basename(src))
            # Avoid name collision
            if os.path.exists(dst):
                base, ext = os.path.splitext(os.path.basename(src))
                dst = os.path.join(quarantine_ds, f"{base}_{moved}{ext}")
            shutil.move(src, dst)
            moved += 1

    print(f"  {name}: Quarantined {moved} corrupted files to {quarantine_ds}")
    return moved


def detect_duplicates_by_hash(name, df, max_images=5000):
    """Detect near-duplicate images using perceptual hashing.
    
    For large datasets (fruits360 has 90k+ images), we sample to keep runtime reasonable.
    """
    valid = df[df["valid"] == True].copy()
    if valid.empty:
        return [], 0

    # For very large datasets, check duplicates per-class to keep it tractable
    total_images = len(valid)
    print(f"  {name}: Checking {total_images} valid images for duplicates...")

    # Group by label and check within each group
    duplicates_found = []
    hashes_computed = 0

    for label, group in valid.groupby("label"):
        if len(group) < 2:
            continue

        hash_map = {}  # hash → path
        for _, row in group.iterrows():
            path = row["path"]
            if not os.path.exists(path):
                continue
            try:
                h = imagehash.phash(Image.open(path))
                hashes_computed += 1

                # Check against existing hashes in this group
                for existing_hash, existing_path in hash_map.items():
                    dist = h - existing_hash
                    if dist <= HASH_DISTANCE_THRESHOLD:
                        duplicates_found.append({
                            "original": existing_path,
                            "duplicate": path,
                            "label": label,
                            "hash_distance": dist,
                        })
                        break
                else:
                    hash_map[h] = path

            except Exception as e:
                pass  # Skip files that can't be hashed

        # Progress indicator for large datasets
        if hashes_computed % 5000 == 0 and hashes_computed > 0:
            print(f"    ... processed {hashes_computed} images so far...")

    print(f"  {name}: Computed {hashes_computed} hashes, found {len(duplicates_found)} duplicates")
    return duplicates_found, hashes_computed


def quarantine_duplicates(name, duplicates):
    """Move duplicate files to quarantine."""
    if not duplicates:
        return 0

    quarantine_ds = os.path.join(QUARANTINE_DIR, f"{name}_duplicates")
    os.makedirs(quarantine_ds, exist_ok=True)

    moved = 0
    for dup in duplicates:
        src = dup["duplicate"]
        if os.path.exists(src):
            dst = os.path.join(quarantine_ds, os.path.basename(src))
            if os.path.exists(dst):
                base, ext = os.path.splitext(os.path.basename(src))
                dst = os.path.join(quarantine_ds, f"{base}_{moved}{ext}")
            shutil.move(src, dst)
            moved += 1

    print(f"  {name}: Quarantined {moved} duplicate files to {quarantine_ds}")
    return moved


if __name__ == "__main__":
    print("="*60)
    print("CORRUPTED & DUPLICATE IMAGE CLEANUP")
    print("="*60)

    summary = {}

    for name in DATASETS:
        csv_path = os.path.join(PROCESSED_DIR, f"audit_{name}.csv")
        if not os.path.exists(csv_path):
            print(f"\nAudit CSV not found for {name}. Run data_audit.py first!")
            continue

        print(f"\n--- Processing {name} ---")
        df = pd.read_csv(csv_path)

        # Step 1: Quarantine corrupted files
        corrupted_count = quarantine_corrupted(name, df)

        # Step 2: Detect and quarantine duplicates
        duplicates, hashes_computed = detect_duplicates_by_hash(name, df)
        duplicate_count = quarantine_duplicates(name, duplicates)

        summary[name] = {
            "corrupted_removed": corrupted_count,
            "duplicates_removed": duplicate_count,
            "total_removed": corrupted_count + duplicate_count,
            "hashes_computed": hashes_computed,
        }

        # Save duplicate log
        if duplicates:
            dup_log_path = os.path.join(PROCESSED_DIR, f"duplicates_{name}.csv")
            pd.DataFrame(duplicates).to_csv(dup_log_path, index=False)
            print(f"  Duplicate log saved to: {dup_log_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("CLEANUP SUMMARY")
    print(f"{'='*60}")
    total_removed = 0
    for name, stats in summary.items():
        print(f"  {name}:")
        print(f"    Corrupted files quarantined: {stats['corrupted_removed']}")
        print(f"    Duplicate files quarantined: {stats['duplicates_removed']}")
        print(f"    Total removed: {stats['total_removed']}")
        total_removed += stats["total_removed"]

    print(f"\n  Grand total files removed: {total_removed}")
    print("\nDone!")
