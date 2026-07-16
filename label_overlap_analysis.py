"""
Day 3 — Task 4: Label Overlap & Naming Inconsistency Analysis
Collects all unique raw labels across all datasets and uses fuzzy matching
to detect near-duplicate labels that likely refer to the same real-world item.
"""

import os
import pandas as pd
from fuzzywuzzy import fuzz
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DATASETS = ["fruits360", "grocery", "fridge_objects"]
FUZZY_THRESHOLD = 75  # Minimum similarity score to flag as potential overlap


def normalize_label(label):
    """Normalize a label for comparison: lowercase, strip, replace separators."""
    label = str(label).lower().strip()
    label = label.replace("-", " ").replace("_", " ")
    return label


def load_all_labels():
    """Load unique labels from each dataset's audit CSV."""
    dataset_labels = {}
    for name in DATASETS:
        csv_path = os.path.join(PROCESSED_DIR, f"audit_{name}.csv")
        if not os.path.exists(csv_path):
            print(f"  Audit CSV not found for {name}, skipping...")
            continue
        df = pd.read_csv(csv_path)
        valid = df[df["valid"] == True]
        labels = sorted(valid["label"].unique().tolist())
        dataset_labels[name] = labels
        print(f"  {name}: {len(labels)} unique labels")
    return dataset_labels


def find_exact_overlaps(dataset_labels):
    """Find labels that appear in multiple datasets (case-insensitive)."""
    # Build a map of normalized_label → [(dataset, original_label)]
    label_map = defaultdict(list)
    for ds_name, labels in dataset_labels.items():
        for label in labels:
            norm = normalize_label(label)
            label_map[norm].append((ds_name, label))

    overlaps = {k: v for k, v in label_map.items() if len(set(ds for ds, _ in v)) > 1}
    return overlaps


def find_fuzzy_overlaps(dataset_labels):
    """Find near-duplicate labels across datasets using fuzzy matching."""
    # Collect all (dataset, label) pairs
    all_pairs = []
    for ds_name, labels in dataset_labels.items():
        for label in labels:
            all_pairs.append((ds_name, label, normalize_label(label)))

    fuzzy_matches = []
    seen = set()

    for i, (ds1, lab1, norm1) in enumerate(all_pairs):
        for j, (ds2, lab2, norm2) in enumerate(all_pairs):
            if i >= j:
                continue
            if norm1 == norm2:
                continue  # Exact matches handled separately

            key = tuple(sorted([norm1, norm2]))
            if key in seen:
                continue

            score = fuzz.ratio(norm1, norm2)
            if score >= FUZZY_THRESHOLD:
                fuzzy_matches.append({
                    "label_1": lab1,
                    "dataset_1": ds1,
                    "label_2": lab2,
                    "dataset_2": ds2,
                    "similarity": score,
                })
                seen.add(key)

    return sorted(fuzzy_matches, key=lambda x: x["similarity"], reverse=True)


def find_within_dataset_variants(dataset_labels):
    """Find labels within the same dataset that are variants of each other."""
    variants = {}
    for ds_name, labels in dataset_labels.items():
        norms = [(label, normalize_label(label)) for label in labels]
        ds_variants = []
        for i, (lab1, norm1) in enumerate(norms):
            for j, (lab2, norm2) in enumerate(norms):
                if i >= j:
                    continue
                # Check if one is a substring of another or very similar
                score = fuzz.ratio(norm1, norm2)
                token_score = fuzz.token_sort_ratio(norm1, norm2)
                if score >= 70 or token_score >= 85:
                    ds_variants.append({
                        "label_1": lab1,
                        "label_2": lab2,
                        "similarity": score,
                        "token_sort_similarity": token_score,
                    })
        if ds_variants:
            variants[ds_name] = sorted(ds_variants, key=lambda x: x["similarity"], reverse=True)
    return variants


if __name__ == "__main__":
    print("="*60)
    print("LABEL OVERLAP & NAMING INCONSISTENCY ANALYSIS")
    print("="*60)

    # Load all labels
    print("\nLoading labels from audit CSVs...")
    dataset_labels = load_all_labels()

    if not dataset_labels:
        print("No datasets found. Run data_audit.py first!")
        exit(1)

    # 1. Exact cross-dataset overlaps
    print(f"\n{'='*60}")
    print("EXACT CROSS-DATASET OVERLAPS (case-insensitive)")
    print(f"{'='*60}")
    exact_overlaps = find_exact_overlaps(dataset_labels)
    if exact_overlaps:
        for norm_label, sources in sorted(exact_overlaps.items()):
            datasets = ", ".join(f"{ds}:'{lab}'" for ds, lab in sources)
            print(f"  '{norm_label}' → appears in: {datasets}")
        print(f"\n  Total exact overlaps: {len(exact_overlaps)}")
    else:
        print("  No exact cross-dataset overlaps found.")

    # 2. Fuzzy cross-dataset overlaps
    print(f"\n{'='*60}")
    print(f"FUZZY CROSS-DATASET OVERLAPS (similarity >= {FUZZY_THRESHOLD}%)")
    print(f"{'='*60}")
    fuzzy_matches = find_fuzzy_overlaps(dataset_labels)
    if fuzzy_matches:
        for m in fuzzy_matches[:50]:  # Show top 50
            print(f"  [{m['similarity']}%] '{m['label_1']}' ({m['dataset_1']}) "
                  f"↔ '{m['label_2']}' ({m['dataset_2']})")
        if len(fuzzy_matches) > 50:
            print(f"  ... and {len(fuzzy_matches) - 50} more")
        print(f"\n  Total fuzzy matches: {len(fuzzy_matches)}")
    else:
        print("  No fuzzy cross-dataset overlaps found.")

    # 3. Within-dataset variants
    print(f"\n{'='*60}")
    print("WITHIN-DATASET LABEL VARIANTS (potential duplicates)")
    print(f"{'='*60}")
    variants = find_within_dataset_variants(dataset_labels)
    if variants:
        for ds_name, ds_variants in variants.items():
            print(f"\n  {ds_name}:")
            for v in ds_variants[:30]:  # Show top 30 per dataset
                print(f"    [{v['similarity']}%] '{v['label_1']}' ↔ '{v['label_2']}'")
            if len(ds_variants) > 30:
                print(f"    ... and {len(ds_variants) - 30} more")
    else:
        print("  No within-dataset variants found.")

    # 4. All unique labels across all datasets
    print(f"\n{'='*60}")
    print("ALL UNIQUE LABELS ACROSS ALL DATASETS")
    print(f"{'='*60}")
    all_labels = set()
    for ds_name, labels in dataset_labels.items():
        for label in labels:
            all_labels.add(normalize_label(label))
    print(f"  Total unique normalized labels: {len(all_labels)}")

    # Save analysis results
    results_path = os.path.join(PROCESSED_DIR, "label_overlap_analysis.csv")
    if fuzzy_matches:
        pd.DataFrame(fuzzy_matches).to_csv(results_path, index=False)
        print(f"\n  Fuzzy match results saved to: {results_path}")

    print("\nDone!")
