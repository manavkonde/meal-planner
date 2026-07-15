# AI Meal Planner — AI-Powered Food Detection & Meal Planning

An AI system that detects ingredients from fridge/pantry photos, estimates nutrition, and generates personalized meal plans using YOLOv8 object detection.

**Status:** Week 1 Complete ✅ | Week 2 Ready for Training

---

## 🎯 Project Status

| Week | Phase | Status | Key Deliverables |
|------|-------|--------|-----------------|
| **Week 1** | Foundation | ✅ COMPLETE | Day 3-5 fully implemented |
| **Week 2** | Model Training | 🚀 READY | Dataset prepared, baseline established |
| **Week 3+** | Integration | ⏳ TODO | Backend, nutrition API, meal planning |

---

## 📅 Week 1 Implementation

### Day 3: Dataset Exploration & Cleaning ✅
- **Status:** Complete (prior completion)
- **Audited 3 datasets:**
  - Fruits-360: 90,380 images, 131 classes
  - Grocery: 5,421 images, 81 classes
  - Fridge Objects: 1,162 images, 8 classes
- **Data Cleaning:**
  - Removed 75,594 duplicate images via perceptual hashing
  - 0 corrupted files detected
- **Unified Taxonomy:** Created 40 canonical fridge/pantry classes
- **Output:** `data/processed/class_mapping.json`, audit CSVs, `docs/day3_summary.md`

### Day 4: YOLO Format Conversion ✅
- **Status:** Complete
- **Dataset Conversion:**
  - Converted 15,371 images to YOLOv8 format
  - Generated normalized YOLO `.txt` label files
  - Applied whole-image bounding boxes for classification-only datasets
- **Train/Val/Test Split:** Stratified 80/10/10 (no data leakage)
  - Train: 12,108 images (78.8%)
  - Val: 1,521 images (9.9%)
  - Test: 1,562 images (10.2%)
- **Output:** 
  - `data/processed/images/{train,val,test}/` (15,371 images)
  - `data/processed/labels/{train,val,test}/` (15,371 label files)
  - `data/processed/data.yaml` (YOLOv8 config)
  - `notebooks/02_yolo_conversion.ipynb` (walkthrough)
  - `docs/day4_summary.md` (detailed report)

### Day 5: Baseline Detection Testing ✅
- **Status:** Complete
- **Pre-trained YOLOv8 Testing:**
  - Ran COCO-pretrained YOLOv8n on 32 real fridge photos
  - Generated 205 total detections (6.4 avg per image)
  - Created annotated images with bounding boxes
- **Key Findings:**
  - **Class Coverage:** Only 12.5% (5/40 classes) in COCO
  - **Accuracy:** ~19.5% correct detections
  - **False Positives:** ~51.2% hallucinated/wrong detections
- **Failure Patterns Identified:**
  1. Missing class categories (87.5% of target items)
  2. Packaging blindness (unaware of container contents)
  3. Hallucinations (donut, cat, table detected in fridges)
  4. Excessive duplicates (same item detected 13x)
  5. Poor confidence calibration
- **Conclusion:** Fine-tuning is **essential** for production use
- **Output:**
  - `backend/modules/baseline_detector.py` (detection script)
  - `results/day5_baseline/` (32 annotated images + CSV/JSON results)
  - `notebooks/03_day5_baseline_testing.ipynb` (analysis notebook)
  - `docs/day5_baseline_summary.md` (comprehensive report)

---

## 📊 Dataset Statistics

### Source Datasets
| Dataset | Images | Classes | Status |
|---------|--------|---------|--------|
| Fruits-360 | 90,380 | 131 | ✅ Processed |
| Grocery | 5,421 | 81 | ✅ Processed |
| Fridge Objects | 1,162 | 8 | ✅ Processed |
| Test Fridges | 32 | — | ✅ Tested |
| **Total** | **96,995** | — | ✅ Ready |

### Unified Taxonomy
- **Target classes:** 40 canonical fridge/pantry items
- **Coverage:** 149/206 raw labels mapped (72.3%)
- **Excluded:** 57 labels (commercial brands, exotic items, non-produce)
- **Classes:** apple, avocado, banana, beef, bell_pepper, blueberry, bread, cabbage, carrot, cauliflower, cherry, corn, cucumber, eggplant, eggs, garlic, ginger, grape, kiwi, lemon, lime, mango, milk, mushroom, onion, orange, peach, pear, pineapple, plum, pomegranate, potato, raspberry, spinach, strawberry, tomato, watermelon, yogurt, zucchini, [+2 more]

### Processed Dataset
- **Total images:** 15,371 (after mapping)
- **Training set:** 12,108 (78.8%)
- **Validation set:** 1,521 (9.9%)
- **Test set:** 1,562 (10.2%)
- **Format:** YOLO-compatible with normalized bounding boxes
- **Status:** ✅ Production-ready for training

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. View Week 1 Results
```bash
# Day 4: YOLO Conversion
jupyter notebook notebooks/02_yolo_conversion.ipynb

# Day 5: Baseline Testing
jupyter notebook notebooks/03_day5_baseline_testing.ipynb
```

### 3. Explore Results
```bash
# Annotated fridge photos with baseline detections
ls results/day5_baseline/annotated_images/

# Detection results
cat results/day5_baseline/baseline_detections.csv

# Model configuration
cat data/processed/data.yaml
```

---

## 📋 Project Structure

```
meal-planner/
├── data/
│   ├── raw/                  # Original datasets (90k+ images)
│   │   ├── fruits360/        # Fruits-360 dataset
│   │   ├── grocery/          # Grocery Store Dataset
│   │   ├── fridge_objects/   # Fridge detection dataset
│   │   └── test_fridges/     # Real-world test photos (32 images)
│   ├── processed/            # Cleaned & YOLO-formatted
│   │   ├── images/           # Split into train/val/test
│   │   ├── labels/           # YOLO .txt label files
│   │   ├── class_mapping.json        # Unified taxonomy
│   │   └── data.yaml                 # YOLOv8 config
│   └── quarantine/           # Duplicate files removed
├── backend/
│   └── modules/
│       ├── yolo_conversion.py        # Day 4: YOLO conversion
│       ├── yolo_validation.py        # Day 4: Validation & viz
│       └── baseline_detector.py      # Day 5: Baseline testing
├── notebooks/
│   ├── 02_yolo_conversion.ipynb      # Day 4 walkthrough
│   └── 03_day5_baseline_testing.ipynb # Day 5 analysis
├── docs/
│   ├── day3_summary.md      # Dataset exploration report
│   ├── day4_summary.md      # YOLO conversion report
│   └── day5_baseline_summary.md  # Baseline testing report
├── results/
│   └── day5_baseline/       # Baseline detection results
│       ├── annotated_images/    # 32 annotated fridges
│       ├── baseline_detections.csv
│       └── baseline_detections.json
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔬 Day 5: Baseline Results Summary

### Pre-trained YOLOv8 Performance
| Metric | Value |
|--------|-------|
| Test images | 32 real fridge photos |
| Total detections | 205 |
| Avg detections/image | 6.4 |
| Correct detections | ~40 (19.5%) |
| False positives | ~105 (51.2%) |
| COCO class coverage | 5/40 (12.5%) |

### Why Fine-tuning is Essential
1. **87.5% of target classes missing** from COCO's 80 classes
2. **51% hallucination rate** (detects non-existent items)
3. **Packaging blindness** (can't recognize contents in containers)
4. **Domain mismatch** (fridge interiors unlike COCO scenes)

### Expected Improvement After Fine-tuning
- Target accuracy: **>80%** (vs 19.5% baseline)
- **4-6x improvement** from baseline
- Production-ready fridge scanning capability

---

## 📈 Week 2 Roadmap

### Days 1-2: Model Fine-tuning
```bash
# Train YOLOv8 on custom dataset
python -c "
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # Medium model
results = model.train(
    data='data/processed/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
"
```

### Day 3: Evaluation & Testing
- Run fine-tuned model on test fridge photos
- Compare before/after detection results
- Generate performance metrics

### Days 4-5: Backend Integration
- Create FastAPI inference endpoint
- Integrate with nutrition API
- Build meal planning logic

---

## 🔧 Technologies Used

- **Python 3.10** — Core language
- **PyTorch / Torch Vision** — Deep learning framework
- **YOLOv8 (Ultralytics)** — Object detection model
- **FastAPI** — Backend framework (Week 2+)
- **Pandas / NumPy** — Data processing
- **OpenCV** — Image processing
- **Jupyter** — Interactive notebooks

---

## 📝 Files & Outputs Generated

### Day 4: YOLO Conversion
- ✅ `backend/modules/yolo_conversion.py` — Main conversion pipeline
- ✅ `backend/modules/yolo_validation.py` — Validation & visualization
- ✅ `notebooks/02_yolo_conversion.ipynb` — Interactive walkthrough
- ✅ `data/processed/images/{train,val,test}/` — 15,371 images
- ✅ `data/processed/labels/{train,val,test}/` — 15,371 label files
- ✅ `data/processed/data.yaml` — YOLOv8 config
- ✅ `docs/day4_summary.md` — Comprehensive report

### Day 5: Baseline Testing
- ✅ `backend/modules/baseline_detector.py` — Detection script
- ✅ `results/day5_baseline/annotated_images/` — 32 annotated photos
- ✅ `results/day5_baseline/baseline_detections.csv` — Results table
- ✅ `results/day5_baseline/baseline_detections.json` — Detailed data
- ✅ `notebooks/03_day5_baseline_testing.ipynb` — Analysis notebook
- ✅ `docs/day5_baseline_summary.md` — Analysis report

---

## 🎯 Portfolio Impact

This Week 1 implementation demonstrates:
1. ✅ **Structured Data Engineering** — 3-dataset merge, cleaning, taxonomy
2. ✅ **ML Pipeline Development** — Format conversion, splitting, validation
3. ✅ **Baseline Establishment** — Pre-trained model evaluation & gap analysis
4. ✅ **Quantified Insights** — 87.5% class coverage gap, 51% hallucination rate
5. ✅ **Documentation Quality** — Comprehensive reports and interactive notebooks

**Interview Narrative:** "I built a complete Week 1 foundation for an AI meal planner, merging 3 datasets into a unified 40-class taxonomy, converting 15,371 images to YOLO format, and establishing a baseline showing why fine-tuning is essential through quantified gap analysis..."

---

## 📞 Support

For detailed information, see:
- Day 3: `docs/day3_summary.md`
- Day 4: `docs/day4_summary.md` and `notebooks/02_yolo_conversion.ipynb`
- Day 5: `docs/day5_baseline_summary.md` and `notebooks/03_day5_baseline_testing.ipynb`

---

**Project Status:** Week 1 ✅ COMPLETE | Week 2 READY FOR TRAINING | [View GitHub Issues →]()
```powershell
py -3.10 -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify installation
```powershell
python verify_env.py
```
Expected output: `All core libraries loaded successfully!`

---

## 📦 Datasets (Day 2)

| Dataset | Folder | Images | Purpose |
|---------|--------|--------|---------|
| [Fruits-360](https://github.com/Horea94/Fruit-Images-Dataset) | `data/raw/fruits360/` | ~90,500 | Fruit/veg classification |
| [Grocery Store](https://github.com/marcusklasson/GroceryStoreDataset) | `data/raw/grocery/` | ~5,700 | Packaged & fresh grocery items |
| [Fridge Objects](https://www.kaggle.com/datasets/surendraallam/refrigerator-contents) | `data/raw/fridge_objects/` | Manual download | Object detection in fridge context |
| Real-world test fridges | `data/raw/test_fridges/` | 30–50 | Real-world evaluation set |

### Download datasets
```powershell
python download_datasets.py    # Fruits-360 + Grocery (auto)
python download_fridge_only.py # Fridge objects (retry if needed)
python download_test_fridges.py # Real-world fridge photos
```

### ⚠️ Fridge Objects Dataset — Manual Download Required
The odFridgeObjects Azure CDN URLs are currently restricted. Download manually:
1. Go to: https://www.kaggle.com/datasets/surendraallam/refrigerator-contents
2. Click **Download** → extract into `data/raw/fridge_objects/`

OR with Kaggle CLI:
```bash
# 1. Place kaggle.json in ~/.kaggle/
pip install kaggle
kaggle datasets download -d surendraallam/refrigerator-contents -p data/raw/fridge_objects --unzip
```

### Verify datasets
```powershell
python verify_datasets.py
```

---

## 🔑 Environment Variables
Create a `.env` file (git-ignored) for API keys:
```
OPENAI_API_KEY=your_key_here
NUTRITION_API_KEY=your_key_here
```

---

## 🚀 Next Steps (Day 3)
- Audit class distributions across datasets
- Standardize naming conventions
- Remove duplicates and corrupted files
- Build unified class taxonomy (`class_mapping.json`, ~30–50 classes)

---

## 📋 Acceptance Criteria (Day 1–2)

- [x] Python 3.10 venv created and activated
- [x] All packages install cleanly
- [x] `verify_env.py` runs with zero exceptions
- [x] Folder structure matches specification
- [x] `.gitignore` committed, first Git commit made
- [x] Fruits-360 downloaded and extracted (90,503 images)
- [x] Grocery Store Dataset downloaded and extracted (5,717 images)
- [~] Fridge Objects Dataset — manual Kaggle download required (see above)
- [x] 30+ real-world fridge test photos in `data/raw/test_fridges/`
