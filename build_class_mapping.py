"""
Day 3 — Task 6: Build Unified Class Taxonomy
Creates class_mapping.json that maps every raw label from all datasets to one
of ~30-50 canonical fridge/pantry ingredient classes.

Labels that don't map to any target class are documented as excluded.
"""

import os
import json
import pandas as pd
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DATASETS = ["fruits360", "grocery", "fridge_objects"]

# ── Target Taxonomy: ~40 common fridge/pantry items ──────────────────────────
TARGET_CLASSES = [
    "apple", "apricot", "avocado", "banana", "beetroot", "bell_pepper",
    "blueberry", "bread", "broccoli", "cabbage", "carrot", "cauliflower",
    "cherry", "corn", "cucumber", "eggplant", "eggs", "garlic", "ginger",
    "grape", "kiwi", "lemon", "lettuce", "lime", "mango", "milk",
    "mushroom", "onion", "orange", "peach", "pear", "pineapple", "plum",
    "pomegranate", "potato", "raspberry", "spinach", "strawberry", "tomato",
    "watermelon", "yogurt", "zucchini",
]

# ── Raw label → target class mapping ─────────────────────────────────────────
# This maps every raw label (lowercased) encountered across all datasets to
# exactly one target class, or to None if excluded.

RAW_TO_TARGET = {
    # ── Fruits-360 labels ────────────────────────────────────────────────────
    # Apples (all varieties → apple)
    "apple braeburn": "apple",
    "apple crimson snow": "apple",
    "apple golden 1": "apple",
    "apple golden 2": "apple",
    "apple golden 3": "apple",
    "apple granny smith": "apple",
    "apple pink lady": "apple",
    "apple red 1": "apple",
    "apple red 2": "apple",
    "apple red 3": "apple",
    "apple red delicious": "apple",
    "apple red yellow 1": "apple",
    "apple red yellow 2": "apple",
    "apricot": "apricot",
    "avocado": "avocado",
    "avocado ripe": "avocado",
    "banana": "banana",
    "banana lady finger": "banana",
    "banana red": "banana",
    "beetroot": "beetroot",
    "blueberry": "blueberry",
    "cactus fruit": None,  # Excluded: not a common fridge item
    "cantaloupe 1": None,  # Excluded: mapped to watermelon would be wrong, not common enough
    "cantaloupe 2": None,
    "carambula": None,     # Excluded: not a common fridge item (star fruit)
    "cauliflower": "cauliflower",
    "cherry 1": "cherry",
    "cherry 2": "cherry",
    "cherry rainier": "cherry",
    "cherry wax black": "cherry",
    "cherry wax red": "cherry",
    "cherry wax yellow": "cherry",
    "chestnut": None,      # Excluded: not a common fridge item
    "clementine": "orange",  # Map to orange (citrus family)
    "cocos": None,         # Excluded: coconut not common in typical fridge
    "corn": "corn",
    "corn husk": "corn",
    "cucumber ripe": "cucumber",
    "cucumber ripe 2": "cucumber",
    "dates": None,         # Excluded: not a common fridge item
    "eggplant": "eggplant",
    "fig": None,           # Excluded: not common enough
    "ginger root": "ginger",
    "granadilla": None,    # Excluded: not common
    "grape blue": "grape",
    "grape pink": "grape",
    "grape white": "grape",
    "grape white 2": "grape",
    "grape white 3": "grape",
    "grape white 4": "grape",
    "grapefruit pink": None,  # Excluded: separate from orange
    "grapefruit white": None,
    "guava": None,         # Excluded: not common in typical fridge
    "hazelnut": None,      # Excluded: nut, not fresh produce
    "huckleberry": None,   # Excluded: rare
    "kaki": None,          # Excluded: persimmon, not common enough
    "kiwi": "kiwi",
    "kohlrabi": None,      # Excluded: not common enough
    "kumquats": None,      # Excluded: not common
    "lemon": "lemon",
    "lemon meyer": "lemon",
    "limes": "lime",
    "lychee": None,        # Excluded: not common enough
    "mandarine": "orange",
    "mango": "mango",
    "mango red": "mango",
    "mangostan": None,     # Excluded: mangosteen, not common
    "maracuja": None,      # Excluded: passion fruit, not common enough
    "melon piel de sapo": None,  # Excluded: specific melon variety
    "mulberry": None,      # Excluded: not common
    "nectarine": "peach",  # Map to peach (very similar)
    "nectarine flat": "peach",
    "nut forest": None,    # Excluded: not fresh produce
    "nut pecan": None,     # Excluded: not fresh produce
    "onion red": "onion",
    "onion red peeled": "onion",
    "onion white": "onion",
    "orange": "orange",
    "papaya": None,        # Excluded: not common enough in typical fridge
    "passion fruit": None,
    "peach": "peach",
    "peach 2": "peach",
    "peach flat": "peach",
    "pear": "pear",
    "pear 2": "pear",
    "pear abate": "pear",
    "pear forelle": "pear",
    "pear kaiser": "pear",
    "pear monster": "pear",
    "pear red": "pear",
    "pear stone": "pear",
    "pear williams": "pear",
    "pepino": None,        # Excluded: not common
    "pepper green": "bell_pepper",
    "pepper orange": "bell_pepper",
    "pepper red": "bell_pepper",
    "pepper yellow": "bell_pepper",
    "physalis": None,      # Excluded: not common
    "physalis with husk": None,
    "pineapple": "pineapple",
    "pineapple mini": "pineapple",
    "pitahaya red": None,  # Excluded: dragon fruit, not common
    "plum": "plum",
    "plum 2": "plum",
    "plum 3": "plum",
    "pomegranate": "pomegranate",
    "pomelo sweetie": None,  # Excluded: not common
    "potato red": "potato",
    "potato red washed": "potato",
    "potato sweet": "potato",
    "potato white": "potato",
    "quince": None,        # Excluded: not common
    "rambutan": None,      # Excluded: not common
    "raspberry": "raspberry",
    "redcurrant": None,    # Excluded: not common
    "salak": None,         # Excluded: snake fruit, not common
    "strawberry": "strawberry",
    "strawberry wedge": "strawberry",
    "tamarillo": None,     # Excluded: not common
    "tangelo": "orange",
    "tomato 1": "tomato",
    "tomato 2": "tomato",
    "tomato 3": "tomato",
    "tomato 4": "tomato",
    "tomato cherry red": "tomato",
    "tomato heart": "tomato",
    "tomato maroon": "tomato",
    "tomato yellow": "tomato",
    "tomato not ripened": "tomato",
    "walnut": None,        # Excluded: nut
    "watermelon": "watermelon",

    # ── Grocery Store Dataset labels ─────────────────────────────────────────
    # Fruits
    "golden-delicious": "apple",
    "granny-smith": "apple",
    "pink-lady": "apple",
    "red-delicious": "apple",
    "royal-gala": "apple",
    "cantaloupe": None,
    "galia-melon": None,
    "honeydew-melon": None,
    "nectarine": "peach",
    "papaya": None,
    "passion-fruit": None,
    "anjou": "pear",
    "conference": "pear",
    "kaiser": "pear",
    "red-grapefruit": None,
    "satsumas": "orange",
    "beef-tomato": "tomato",
    "regular-tomato": "tomato",
    "vine-tomato": "tomato",
    # Vegetables
    "asparagus": None,     # Excluded: not common enough for this taxonomy
    "aubergine": "eggplant",
    "cabbage": "cabbage",
    "carrots": "carrot",
    "cucumber": "cucumber",
    "garlic": "garlic",
    "ginger": "ginger",
    "leek": None,          # Excluded: not common enough
    "brown-cap-mushroom": "mushroom",
    "yellow-onion": "onion",
    "green-bell-pepper": "bell_pepper",
    "orange-bell-pepper": "bell_pepper",
    "red-bell-pepper": "bell_pepper",
    "yellow-bell-pepper": "bell_pepper",
    "floury-potato": "potato",
    "solid-potato": "potato",
    "sweet-potato": "potato",
    "red-beet": "beetroot",
    "zucchini": "zucchini",
    # Packages
    "bravo-apple-juice": None,       # Excluded: packaged product
    "bravo-orange-juice": None,
    "god-morgon-apple-juice": None,
    "god-morgon-orange-juice": None,
    "god-morgon-orange-red-grapefruit-juice": None,
    "god-morgon-red-grapefruit-juice": None,
    "tropicana-apple-juice": None,
    "tropicana-golden-grapefruit": None,
    "tropicana-juice-smooth": None,
    "tropicana-mandarin-morning": None,
    "arla-ecological-medium-fat-milk": "milk",
    "arla-lactose-medium-fat-milk": "milk",
    "arla-medium-fat-milk": "milk",
    "arla-standard-milk": "milk",
    "garant-ecological-medium-fat-milk": "milk",
    "garant-ecological-standard-milk": "milk",
    "oatly-natural-oatghurt": "yogurt",
    "oatly-oat-milk": "milk",
    "arla-ecological-sour-cream": None,    # Excluded: sour cream not in target
    "arla-sour-cream": None,
    "arla-sour-milk": "milk",
    "alpro-blueberry-soyghurt": "yogurt",
    "alpro-vanilla-soyghurt": "yogurt",
    "alpro-fresh-soy-milk": "milk",
    "alpro-shelf-soy-milk": "milk",
    "arla-mild-vanilla-yoghurt": "yogurt",
    "arla-natural-mild-low-fat-yoghurt": "yogurt",
    "arla-natural-yoghurt": "yogurt",
    "valio-vanilla-yoghurt": "yogurt",
    "yoggi-strawberry-yoghurt": "yogurt",
    "yoggi-vanilla-yoghurt": "yogurt",

    # ── Fridge Object Detection Dataset labels ───────────────────────────────
    "Banana": "banana",
    "Bread": "bread",
    "Eggs": "eggs",
    "Milk": "milk",
    "Mixed": None,         # Excluded: mixed items don't map to a single class
    "Potato": "potato",
    "Spinach": "spinach",
    "Tomato": "tomato",

    # ── Grocery dataset coarse-level labels (in case they appear) ────────────
    "Apple": "apple",
    "Avocado": "avocado",
    "Banana": "banana",
    "Kiwi": "kiwi",
    "Lemon": "lemon",
    "Lime": "lime",
    "Mango": "mango",
    "Melon": None,
    "Orange": "orange",
    "Peach": "peach",
    "Pear": "pear",
    "Pineapple": "pineapple",
    "Plum": "plum",
    "Pomegranate": "pomegranate",
    "Juice": None,
    "Milk": "milk",
    "Yoghurt": "yogurt",
    "Asparagus": None,
    "Aubergine": "eggplant",
    "Cabbage": "cabbage",
    "Carrots": "carrot",
    "Cucumber": "cucumber",
    "Garlic": "garlic",
    "Ginger": "ginger",
    "Leek": None,
    "Mushroom": "mushroom",
    "Onion": "onion",
    "Pepper": "bell_pepper",
    "Potato": "potato",
    "Red-Beet": "beetroot",
    "Tomato": "tomato",
    "Zucchini": "zucchini",

    # Grocery sub-categories that might appear
    "Fruit": None,         # Too broad
    "Vegetables": None,    # Too broad
    "Packages": None,      # Too broad
}


def collect_all_raw_labels():
    """Collect every unique raw label from audit CSVs."""
    all_labels = defaultdict(set)  # label → set of datasets it appears in
    for name in DATASETS:
        csv_path = os.path.join(PROCESSED_DIR, f"audit_{name}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        valid = df[df["valid"] == True]
        for label in valid["label"].unique():
            all_labels[str(label)].add(name)
    return all_labels


def build_mapping(all_labels):
    """Build the complete mapping, handling unmapped labels."""
    mapping = {}
    excluded = {}
    unmapped = {}

    for raw_label, datasets in all_labels.items():
        # Try exact match first
        if raw_label in RAW_TO_TARGET:
            target = RAW_TO_TARGET[raw_label]
            if target is not None:
                mapping[raw_label] = target
            else:
                excluded[raw_label] = list(datasets)
            continue

        # Try lowercase match
        lower = raw_label.lower()
        if lower in RAW_TO_TARGET:
            target = RAW_TO_TARGET[lower]
            if target is not None:
                mapping[raw_label] = target
            else:
                excluded[raw_label] = list(datasets)
            continue

        # Try normalized match (replace separators)
        normalized = lower.replace("-", " ").replace("_", " ").strip()
        found = False
        for key, target in RAW_TO_TARGET.items():
            key_norm = key.lower().replace("-", " ").replace("_", " ").strip()
            if key_norm == normalized:
                if target is not None:
                    mapping[raw_label] = target
                else:
                    excluded[raw_label] = list(datasets)
                found = True
                break

        if not found:
            unmapped[raw_label] = list(datasets)

    return mapping, excluded, unmapped


if __name__ == "__main__":
    print("="*60)
    print("BUILDING UNIFIED CLASS TAXONOMY")
    print("="*60)

    # Collect all raw labels
    print("\nCollecting raw labels from audit CSVs...")
    all_labels = collect_all_raw_labels()
    print(f"  Total unique raw labels found: {len(all_labels)}")

    # Build the mapping
    mapping, excluded, unmapped = build_mapping(all_labels)

    # Report
    target_classes_used = sorted(set(mapping.values()))

    print(f"\n  Mapped labels: {len(mapping)}")
    print(f"  Excluded labels: {len(excluded)}")
    print(f"  Unmapped labels: {len(unmapped)}")
    print(f"  Target classes in use: {len(target_classes_used)}")

    print(f"\n{'='*60}")
    print("TARGET CLASSES IN USE")
    print(f"{'='*60}")
    for i, cls in enumerate(target_classes_used, 1):
        # Count how many raw labels map to this class
        raw_count = sum(1 for v in mapping.values() if v == cls)
        print(f"  {i:2d}. {cls} (← {raw_count} raw labels)")

    print(f"\n{'='*60}")
    print("EXCLUDED LABELS (intentionally not mapped)")
    print(f"{'='*60}")
    for label, datasets in sorted(excluded.items()):
        print(f"  {label} (from {', '.join(datasets)})")

    if unmapped:
        print(f"\n{'='*60}")
        print("⚠️ UNMAPPED LABELS (not found in mapping)")
        print(f"{'='*60}")
        for label, datasets in sorted(unmapped.items()):
            print(f"  {label} (from {', '.join(datasets)})")

    # Save class_mapping.json
    output_path = os.path.join(PROCESSED_DIR, "class_mapping.json")
    output = {
        "mapping": mapping,
        "target_classes": target_classes_used,
        "excluded_labels": {k: v for k, v in sorted(excluded.items())},
        "unmapped_labels": {k: v for k, v in sorted(unmapped.items())},
        "metadata": {
            "total_raw_labels": len(all_labels),
            "total_mapped": len(mapping),
            "total_excluded": len(excluded),
            "total_unmapped": len(unmapped),
            "total_target_classes": len(target_classes_used),
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ class_mapping.json saved to: {output_path}")

    # Also save a flat version for easy use in training pipelines
    flat_path = os.path.join(PROCESSED_DIR, "class_mapping_flat.json")
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Flat mapping saved to: {flat_path}")
    print(f"\n  Total target classes: {len(target_classes_used)}")
    print(f"  Classes: {', '.join(target_classes_used)}")
    print("\nDone!")
