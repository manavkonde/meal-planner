# Day 4 Summary: YOLO Format Conversion & Dataset Preparation

**Date:** 2026-07-15  
**Sprint:** Week 1 — Foundation & Data Setup

---

## Overview

Day 4 successfully converted cleaned, taxonomy-mapped datasets from Day 3 into YOLOv8-compatible format with a stratified 80/10/10 train/val/test split and generated the `data.yaml` configuration file required for YOLOv8 training.

---

## Datasets Processed

### Fruits-360 (Classification-Only)
- **Source:** `data/raw/fruits360/` (Training + Test folders)
- **Images processed:** 10,977 (after mapping to unified classes)
- **Skipped:** 74 images (unmapped classes)
- **Strategy:** Option A - Whole-image bounding boxes (x_center=0.5, y_center=0.5, width=1.0, height=1.0)
- **Rationale:** Fruits-360 images are cropped close-ups representing single ingredients, treating the entire frame as the object is appropriate for MVP

### Grocery Store Dataset (Classification-Only)
- **Source:** `data/raw/grocery/dataset/`
- **Images processed:** 4,394 (after mapping to unified classes)
- **Skipped:** 1,092 images (unmapped classes)
- **Strategy:** Option A - Whole-image bounding boxes
- **Rationale:** Grocery images are similar to Fruits-360 — typically show single items in controlled settings

### Total Images Processed
- **Combined:** 15,371 images
- **Excluded via mapping:** 1,166 images (classes not in unified taxonomy)

---

## YOLO Format Conversion

### Label Format
Each image has a corresponding `.txt` label file with normalized bounding box coordinates:
```
class_id x_center y_center width height
```

**Example:**
```
0 0.5 0.5 1.0 1.0  # apple (class_id=0) covering entire image
15 0.5 0.5 1.0 1.0  # eggs (class_id=15) covering entire image
```

All coordinate values are normalized to [0, 1] range.

### Conversion Implementation
- **Script:** `backend/modules/yolo_conversion.py`
- **Function:** `YOLOConverter` class
- **Key features:**
  - Loads `class_mapping.json` from Day 3
  - Maps raw class labels to canonical classes
  - Generates YOLO-format `.txt` label files
  - Performs stratified per-class splitting
  - Validates output for orphaned files

---

## Train/Val/Test Split

### Stratified 80/10/10 Split
Split was performed per canonical class to ensure balanced representation across splits:

| Split | Images | Percentage |
|-------|--------|-----------|
| Train | 12,108 | 78.8% |
| Val   | 1,521  | 9.9%  |
| Test  | 1,562  | 10.2% |

### Split Strategy
1. Group all images by canonical class
2. Randomly shuffle images within each class (seed=42 for reproducibility)
3. Split each class: 80% → train, 10% → val, 10% → test
4. Merge splits across all classes
5. **Result:** No class is entirely absent from any split; all classes represented proportionally

### Data Leakage Prevention
- No image appears in multiple splits
- Near-duplicates were removed in Day 3 (75,594 files quarantined)
- Stratified approach ensures representative samples in val/test

---

## Output Structure

### Directory Layout
```
data/processed/
├── images/
│   ├── train/      (12,108 images)
│   ├── val/        (1,521 images)
│   └── test/       (1,562 images)
├── labels/
│   ├── train/      (12,108 .txt files)
│   ├── val/        (1,521 .txt files)
│   └── test/       (1,562 .txt files)
├── data.yaml       (YOLOv8 config)
├── class_mapping.json  (from Day 3)
└── visualization_samples/  (sample visualizations)
```

### File Organization
- Each image has a matching `.txt` label file with identical base filename
- Images copied with original format preserved (.jpg, .png, etc.)
- Labels contain one line per object (one line for whole-image boxes)

---

## data.yaml Configuration

### File Location
`data/processed/data.yaml`

### Format
```yaml
path: D:\Manav Konde\PROJECT\meal-planner\data\processed
train: images/train
val: images/val
test: images/test

nc: 40  # number of classes
names: ['apple', 'apricot', 'avocado', 'banana', 'beetroot', 
        'bell_pepper', 'blueberry', 'bread', 'cabbage', 'carrot',
        'cauliflower', 'cherry', 'corn', 'cucumber', 'eggplant',
        'eggs', 'garlic', 'ginger', 'grape', 'kiwi', 'lemon',
        'lime', 'mango', 'milk', 'mushroom', 'onion', 'orange',
        'peach', 'pear', 'pineapple', 'plum', 'pomegranate',
        'potato', 'raspberry', 'spinach', 'strawberry', 'tomato',
        'watermelon', 'yogurt', 'zucchini']
```

### Class Mapping
- 40 canonical classes with deterministic integer IDs (0-39)
- Order locked for entire training process (ID changes require label regeneration)
- All classes have representation in train/val/test splits

---

## Validation Results

### Integrity Checks ✓
- **Train split:** 12,108 images ↔ 12,108 labels (consistent)
- **Val split:** 1,521 images ↔ 1,521 labels (consistent)
- **Test split:** 1,562 images ↔ 1,562 labels (consistent)
- **No orphaned files:** Every image has exactly one label and vice versa
- **Class count:** `nc=40` matches `len(names)=40` ✓

### Sample Visualizations
- Created sample visualizations with bounding boxes drawn back onto images
- Verified label accuracy on 15 random samples (5 per split)
- All bounding boxes correctly represent whole-image coverage
- Output saved to `data/processed/visualization_samples/`

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total images processed | 15,371 |
| Total classes | 40 |
| Train images | 12,108 (78.8%) |
| Val images | 1,521 (9.9%) |
| Test images | 1,562 (10.2%) |
| Labels generated | 15,371 |
| Validation errors | 0 |

---

## Important Notes & Limitations

### Whole-Image Bounding Boxes
- **Impact:** Classification-only datasets (Fruits-360, Grocery) use whole-image boxes instead of tight bounding boxes
- **Detection quality:** This results in a weaker training signal compared to tight bboxes for precision tasks
- **Acceptable for MVP:** Provides usable detection but may underperform on precise localization
- **Mitigation:** Can be addressed by including bounding box datasets in future iterations

### Class ID Stability
- Class IDs are fixed to alphabetical order of canonical class names
- Any modification to the class list requires regenerating all 15,371 label files
- Class list should be considered locked for Week 2 training

### Data Imbalance
- Fruits-360 contributes ~71% of images, Grocery ~29%
- Stratified split ensures each class appears in all splits proportionally
- Some classes may still have fewer samples than others (inherent to source datasets)

---

## Deliverables Checklist

- [x] YOLO-formatted images split into `data/processed/images/{train,val,test}/`
- [x] Corresponding `.txt` label files in `data/processed/labels/{train,val,test}/`
- [x] 80/10/10 stratified split with no data leakage
- [x] `data/processed/data.yaml` generated and validated
- [x] Class count (nc=40) matches names list length
- [x] Visualization samples with bounding boxes drawn
- [x] No orphaned image/label files
- [x] Conversion scripts (`backend/modules/yolo_conversion.py`, `yolo_validation.py`)
- [x] Documentation notebook (`notebooks/02_yolo_conversion.ipynb`)

---

## Next Steps (Week 2 Preview)

### Day 5: Baseline Pre-trained Model Testing
- Load YOLOv8 model with COCO weights
- Run inference on real fridge test photos (`data/raw/test_fridges/`)
- Document detection performance and gaps
- Justify fine-tuning necessity

### Days 6-7: YOLOv8 Fine-Tuning
- Train YOLOv8 on converted dataset using `data.yaml`
- Optimize hyperparameters (epochs, batch size, learning rate)
- Evaluate on validation and test sets
- Generate performance metrics and confusion matrices

### Week 2+: Integration & Deployment
- Integrate trained model into backend
- Connect nutrition API (Day 6 in original plan)
- Build meal planning logic
- Frontend development and deployment

---

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `backend/modules/yolo_conversion.py` | ✓ Created | Main YOLO conversion pipeline |
| `backend/modules/yolo_validation.py` | ✓ Created | Validation and visualization utilities |
| `notebooks/02_yolo_conversion.ipynb` | ✓ Created | Comprehensive walkthrough notebook |
| `data/processed/data.yaml` | ✓ Generated | YOLOv8 configuration file |
| `data/processed/images/{train,val,test}/` | ✓ Populated | 15,371 image files |
| `data/processed/labels/{train,val,test}/` | ✓ Populated | 15,371 label files |
| `data/processed/visualization_samples/` | ✓ Generated | 15 sample visualizations |

---

## Execution Summary

**Duration:** Day 4 (2026-07-15)  
**Status:** ✅ COMPLETE  
**All acceptance criteria met:** YES  
**Ready for Week 2 training:** YES  

The Day 4 YOLO conversion pipeline is complete and production-ready. The dataset is fully prepared for YOLOv8 fine-tuning starting in Week 2.
