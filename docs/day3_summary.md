# Day 3 Summary: Dataset Exploration & Cleaning

This document summarizes the findings from the dataset exploration, cleaning, and standardization processes executed during Day 3. 

## 1. Data Audit Results
We audited three primary datasets: `fruits360`, `grocery`, and `fridge_objects` (the `test_fridges` dataset was intentionally excluded as it will be used for final validation). 

- **fruits360**: 90,380 images across 131 classes.
- **grocery**: 5,421 images across 81 classes.
- **fridge_objects**: 1,162 images across 8 classes.
- **Data Integrity**: **0 corrupted or unreadable images** were found in the raw downloaded datasets.

## 2. Class Distribution Analysis
- **fruits360**: Average of ~690 images per class. Two classes were severely over-represented (>1000 images): `Grape Blue` (1312) and `Plum 3` (1204). No severely under-represented classes (<20 images) were detected.
- **grocery**: Average of ~67 images per class. The dataset is mostly well-balanced without severe outliers.
- **fridge_objects**: Smallest dataset with exactly 150 images for most classes (except `Mixed` with 114). Perfectly balanced.

## 3. Label Overlap & Naming Inconsistency
A fuzzy matching and exact overlap analysis was performed across the 206 unique raw labels found in the datasets.
- **Exact Overlaps**: 14 labels overlapped exactly across multiple datasets (e.g., "mango", "peach", "watermelon", "orange").
- **Fuzzy Overlaps**: 81 near-match cross-dataset label variants were found (e.g., `Limes` ↔ `Lime`, `Apple Granny Smith` ↔ `Granny-Smith`).
- **Within-Dataset Inconsistency**: Dozens of highly similar labels existed within the same datasets (e.g. `Apple Red Yellow 1` vs `Apple Red Yellow 2` in fruits360, `Green-Bell-Pepper` vs `Red-Bell-Pepper` in grocery).

These overlaps highlighted the absolute necessity of building a unified taxonomy to merge these varying naming conventions.

## 4. Duplicate Image Removal
Using perceptual hashing (`imagehash` phash, threshold=5) to detect nearly identical images:
- **fruits360**: 75,481 near-duplicate images detected and quarantined. This massive number is expected because fruits360 was created by rotating fruits on a motorized turntable, resulting in many contiguous frames being virtually identical. Removing these prevents severe overfitting.
- **grocery**: 16 duplicates detected and quarantined.
- **fridge_objects**: 97 duplicates detected and quarantined.

A grand total of **75,594 files** were safely moved to `data/quarantine/` to prevent training leakage and bias, while preserving the original `data/raw/` structures untouched as requested in the PRD.

## 5. Unified Class Taxonomy
We mapped the 206 unique raw labels down into a **unified taxonomy of 40 canonical fridge/pantry item classes** (e.g. `apple`, `avocado`, `bell_pepper`, `eggs`, `milk`, `tomato`, `yogurt`, etc.).

- **149 raw labels** were successfully mapped to one of the 40 canonical classes.
- **57 raw labels** were intentionally excluded. These included overly specific commercial juice brands (`Bravo-Apple-Juice`), rare exotic fruits (`Cactus fruit`, `Salak`), and non-produce items (`Chestnut`, `Walnut`) that don't fit the general-purpose fridge scanner use case.
- The mapping has been saved to `data/processed/class_mapping.json` (and `class_mapping_flat.json`), establishing the ground truth vocabulary for Day 4 annotation conversion.
