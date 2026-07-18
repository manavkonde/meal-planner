# AI Meal Planner from Fridge Photos

Detect ingredients from fridge or pantry photos, map detections to nutrition data, and use that structured context to generate meal plans.

**Status:** Week 1 complete. The project has a cleaned 40-class taxonomy, a YOLOv8-ready dataset, baseline detection results, and an offline nutrition database.

## Pipeline

1. **Ingredient Detection:** Fine-tune YOLOv8 on the unified fridge/pantry taxonomy.
2. **Nutrition Mapping:** Join detected classes with `data/nutrition_db.csv`.
3. **Meal Plan Generation:** Use detected ingredients plus nutrition context for structured meal suggestions.

## Week 1 Deliverables

| Day | Result | Key files |
|---|---|---|
| Day 3 | Dataset audit, duplicate cleanup, unified 40-class taxonomy | `data/processed/class_mapping.json`, `docs/day3_summary.md` |
| Day 4 | YOLOv8-format dataset with train/val/test split | `data/processed/data.yaml`, `backend/modules/yolo_conversion.py` |
| Day 5 | COCO-pretrained YOLOv8 baseline on real fridge photos | `backend/modules/baseline_detector.py`, `docs/day5_baseline_summary.md` |
| Day 6 | USDA-compatible nutrition lookup and offline cache | `backend/modules/nutrition_lookup.py`, `data/nutrition_db.csv` |
| Day 7 | Repository documentation and Week 1 wrap-up | `docs/week1_summary.md`, `README.md` |

## Project Structure

```text
meal-planner/
├── backend/
│   └── modules/
│       ├── baseline_detector.py
│       ├── nutrition_lookup.py
│       ├── yolo_conversion.py
│       └── yolo_validation.py
├── data/
│   ├── nutrition_db.csv
│   ├── processed/
│   │   ├── class_mapping.json
│   │   ├── data.yaml
│   │   ├── images/
│   │   └── labels/
│   └── raw/
├── docs/
│   ├── day3_summary.md
│   ├── day4_summary.md
│   ├── day5_baseline_summary.md
│   └── week1_summary.md
├── notebooks/
├── results/
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```powershell
py -3.10 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create a local `.env` file for API keys. The file is ignored by Git.

```text
USDA_API_KEY=your_usda_fooddata_central_key_here
```

## Nutrition Data

The project uses `data/nutrition_db.csv` as the offline source of truth for calories and macros per 100g. This keeps demos and later model integration independent from live API calls.

To refresh the cache from USDA FoodData Central after adding a key:

```powershell
python backend\modules\nutrition_lookup.py --validate
```

To validate the existing offline cache without calling the API:

```powershell
python backend\modules\nutrition_lookup.py --no-api --validate
```

## Dataset and Baseline

The processed dataset is configured at `data/processed/data.yaml`:

- 40 target classes
- 12,108 train images
- 1,521 validation images
- 1,562 test images

Day 5 baseline testing used `yolov8n.pt` on 32 real fridge photos. COCO covered only 5 of the 40 target classes and produced a high false-positive rate, which justifies Week 2 fine-tuning.

## Week 2 Preview

Week 2 starts with YOLOv8 fine-tuning against `data/processed/data.yaml`, then compares fine-tuned results against the Day 5 baseline on the same real fridge images.

## Week 2 Training

Day 8/9 fine-tuning starts from local COCO-pretrained nano weights and writes Ultralytics artifacts to `models/checkpoints/yolov8n_fridge_v1/`.

### Day 11 inference wrapper

A reusable inference module is available at `backend/modules/detect_ingredients.py`.

```python
from backend.modules.detect_ingredients import detect_ingredients

results = detect_ingredients("data/raw/test_fridges/fridge_003.jpg")
```

The function returns a list of dictionaries in the form `{item, confidence, bbox}` and raises `FileNotFoundError` for missing images.

Run the Day 11 tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_detect_ingredients.py
```

Check data readiness without starting training:

```powershell
python train_yolov8_fridge.py --dry-run
```

Launch the first training run:

```powershell
python train_yolov8_fridge.py
```

Review Day 9 progress and label the best checkpoint:

```powershell
python monitor_yolov8_training.py --copy-best
```

Training decisions and handoff notes are tracked in `docs/week2_training_log.md`.
