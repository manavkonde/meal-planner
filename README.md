# AI Meal Planner — Project Scaffold

**Week 1 | Day 1–2 Foundation**

An AI system that detects ingredients from fridge/pantry photos, estimates nutrition, and generates a personalized meal plan.

---

## 🗂️ Project Structure

```
meal-planner/
├── data/
│   ├── raw/                  # Untouched downloaded datasets (git-ignored)
│   │   ├── fruits360/        # Fruits-360 — 90,503 images, 131 classes
│   │   ├── grocery/          # Grocery Store Dataset — 5,717 images, 81 classes
│   │   ├── fridge_objects/   # Fridge Object Detection dataset (see note below)
│   │   └── test_fridges/     # 30-50 real-world fridge photos (test set)
│   ├── processed/            # Cleaned, split, YOLO-formatted (Day 3-4)
│   └── nutrition_db.csv      # Nutrition database (Day 6)
├── models/
│   ├── checkpoints/          # Saved model weights
│   └── configs/              # Training config files
├── backend/
│   ├── main.py               # FastAPI entry point
│   └── modules/              # Route handlers, services
├── frontend/                 # UI (Week 3+)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── tests/
├── venv/                     # Python 3.10 virtual environment (git-ignored)
├── requirements.txt          # All dependencies
├── verify_env.py             # Verify core library imports
├── download_datasets.py      # Download Fruits-360, Grocery, Fridge datasets
├── download_fridge_only.py   # Retry fridge dataset download individually
├── download_test_fridges.py  # Download 30-50 real-world fridge test photos
├── verify_datasets.py        # Verify dataset integrity (image counts + spot check)
├── .env                      # API keys (git-ignored)
└── .gitignore
```

---

## ⚙️ Setup (Day 1)

### Prerequisites
- Python 3.10.x
- Git

### 1. Create virtual environment
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
