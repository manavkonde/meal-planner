# Week 1 Summary: AI Meal Planner from Fridge Photos

**Sprint:** Week 1 - Foundation and Data Setup  
**Status:** Complete  
**Next phase:** Week 2 YOLOv8 fine-tuning

## Executive Summary

Week 1 established the data and evaluation foundation for the AI Meal Planner. The project now has a unified 40-class fridge/pantry taxonomy, a YOLOv8-compatible processed dataset, a documented pre-trained YOLO baseline, and a local nutrition database that covers every target class.

The most important Week 1 conclusion is that fine-tuning is required. The COCO-pretrained YOLOv8 baseline covered only 5 of the 40 target classes and produced many fridge-specific false positives.

## Day-by-Day Recap

| Day | Completed work | Outputs |
|---|---|---|
| Day 1-2 | Environment setup, folder scaffolding, raw dataset collection, real fridge test photos | `requirements.txt`, `verify_env.py`, `verify_datasets.py`, `data/raw/` |
| Day 3 | Audited source datasets, removed duplicates, created unified taxonomy | `data/processed/class_mapping.json`, audit CSVs, `docs/day3_summary.md` |
| Day 4 | Converted mapped data into YOLOv8 format and created train/val/test split | `data/processed/data.yaml`, `backend/modules/yolo_conversion.py`, `docs/day4_summary.md` |
| Day 5 | Ran COCO-pretrained YOLOv8 baseline on real fridge images | `results/day5_baseline/`, `backend/modules/baseline_detector.py`, `docs/day5_baseline_summary.md` |
| Day 6 | Added nutrition lookup pipeline and offline nutrition cache | `backend/modules/nutrition_lookup.py`, `data/nutrition_db.csv` |
| Day 7 | Cleaned README and consolidated Week 1 handoff notes | `README.md`, `docs/week1_summary.md` |

## Key Decisions

### Unified Taxonomy

The final taxonomy contains 40 classes:

`apple`, `apricot`, `avocado`, `banana`, `beetroot`, `bell_pepper`, `blueberry`, `bread`, `cabbage`, `carrot`, `cauliflower`, `cherry`, `corn`, `cucumber`, `eggplant`, `eggs`, `garlic`, `ginger`, `grape`, `kiwi`, `lemon`, `lime`, `mango`, `milk`, `mushroom`, `onion`, `orange`, `peach`, `pear`, `pineapple`, `plum`, `pomegranate`, `potato`, `raspberry`, `spinach`, `strawberry`, `tomato`, `watermelon`, `yogurt`, `zucchini`.

This list balances common fridge/pantry usefulness with realistic dataset coverage.

### Dataset Inclusion and Exclusion

The project combines Fruits-360, Grocery Store Dataset, and Fridge Objects. Classification-style images were converted to YOLO format using whole-image bounding boxes. This is imperfect for object localization, but it provides useful class signal for Week 2 transfer learning.

Exotic fruits, commercial juice products, and weakly relevant labels were excluded from the target taxonomy to keep the MVP focused on common meal-planning ingredients.

### Nutrition Source

USDA FoodData Central is the primary live source. `data/nutrition_db.csv` is the offline source of truth for later pipeline stages, with all values expressed per 100g edible portion. The local cache avoids rate limits and keeps demos stable without an API key.

## Baseline Results

The Day 5 baseline used COCO-pretrained `yolov8n.pt` on 32 real fridge photos.

| Metric | Result |
|---|---|
| Test images | 32 |
| Total detections | 205 |
| Average detections per image | 6.4 |
| Correct food detections | About 19.5% |
| False positives | About 51.2% |
| COCO overlap with taxonomy | 5/40 classes |

COCO overlaps with `apple`, `banana`, `broccoli`, `carrot`, and `orange`, but the project taxonomy does not currently include `broccoli`; the practical project overlap is 4 direct classes plus related produce behavior. Critical missing classes include `eggs`, `milk`, `tomato`, `spinach`, `potato`, `onion`, `bread`, and `yogurt`.

Recurring failures:

- Missing target classes for common fridge staples.
- Packaging blindness for cartons, containers, and wrapped items.
- Hallucinated COCO classes in fridge scenes.
- Duplicate detections on the same item.
- Poor confidence calibration in cluttered, low-light images.

## Nutrition Coverage

`data/nutrition_db.csv` contains one row for each of the 40 target classes. Each row includes:

- `class_name`
- `description`
- `calories`
- `protein`
- `carbs`
- `fat`
- `source`
- `notes`

Manual fallback values were filled for all classes so downstream work can proceed before a USDA API key is configured. The script in `backend/modules/nutrition_lookup.py` can refresh the cache from USDA FoodData Central when `USDA_API_KEY` is available.

## Open Issues and Risks

- Whole-image boxes from classification datasets may teach weak localization.
- Real fridge images are more cluttered and packaged than many training examples.
- Some nutrition values are generic per-100g estimates, not brand-specific or portion-aware.
- The model still needs fine-tuning and a before/after evaluation against the Day 5 baseline.
- Large raw and processed datasets should stay out of Git unless a dedicated data storage strategy is chosen.

## Week 2 Readiness

Week 2 can begin with YOLOv8 fine-tuning because the required inputs are present:

- `data/processed/data.yaml`
- `data/processed/class_mapping.json`
- `data/nutrition_db.csv`
- `results/day5_baseline/`
- `docs/day5_baseline_summary.md`

Recommended next step:

```powershell
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print(model.names)"
```

Then start training with `data/processed/data.yaml` once GPU/runtime settings are confirmed.
