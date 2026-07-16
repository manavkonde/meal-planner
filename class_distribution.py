"""
Day 3 — Task 3: Class Distribution Analysis
Loads audit CSVs and analyzes class distribution for each dataset.
Flags under-represented (<20 images) and over-represented classes.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DATASETS = ["fruits360", "grocery", "fridge_objects"]
UNDER_THRESHOLD = 20   # classes with fewer images than this are flagged
OVER_THRESHOLD = 1000  # classes with more images than this are flagged


def analyze_distribution(name, df):
    """Analyze and print class distribution for a dataset."""
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION: {name}")
    print(f"{'='*60}")

    valid = df[df["valid"] == True].copy()
    if valid.empty:
        print("  No valid images found!")
        return

    counts = valid["label"].value_counts()
    total_classes = len(counts)
    total_images = len(valid)

    print(f"\n  Total classes: {total_classes}")
    print(f"  Total valid images: {total_images}")
    print(f"  Images per class — min: {counts.min()}, max: {counts.max()}, "
          f"mean: {counts.mean():.1f}, median: {counts.median():.1f}")

    # Print full distribution
    print(f"\n  Full class distribution (label: count):")
    for label, count in counts.items():
        marker = ""
        if count < UNDER_THRESHOLD:
            marker = " ⚠️ UNDER-REPRESENTED"
        elif count > OVER_THRESHOLD:
            marker = " ⚠️ OVER-REPRESENTED"
        print(f"    {label}: {count}{marker}")

    # Summary of flagged classes
    under = counts[counts < UNDER_THRESHOLD]
    over = counts[counts > OVER_THRESHOLD]

    if len(under) > 0:
        print(f"\n  ⚠️  Under-represented classes (<{UNDER_THRESHOLD} images): {len(under)}")
        for label, count in under.items():
            print(f"    - {label}: {count}")

    if len(over) > 0:
        print(f"\n  ⚠️  Over-represented classes (>{OVER_THRESHOLD} images): {len(over)}")
        for label, count in over.items():
            print(f"    - {label}: {count}")

    if len(under) == 0 and len(over) == 0:
        print(f"\n  ✅ No severely imbalanced classes detected.")

    return counts


if __name__ == "__main__":
    all_distributions = {}

    for name in DATASETS:
        csv_path = os.path.join(PROCESSED_DIR, f"audit_{name}.csv")
        if not os.path.exists(csv_path):
            print(f"Audit CSV not found for {name}: {csv_path}")
            print("Run data_audit.py first!")
            continue

        df = pd.read_csv(csv_path)
        counts = analyze_distribution(name, df)
        if counts is not None:
            all_distributions[name] = counts

    # Cross-dataset summary
    print(f"\n{'='*60}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    for name, counts in all_distributions.items():
        print(f"  {name}: {len(counts)} classes, {counts.sum()} images")

    print("\nDone!")
