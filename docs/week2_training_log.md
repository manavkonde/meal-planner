# Week 2 Training Log

## Day 8 - YOLOv8 Nano Fine-Tuning Setup

**Planned run:** `yolov8n_fridge_v1`  
**Base weights:** `yolov8n.pt`  
**Dataset:** `data/processed/data.yaml`  
**Output directory:** `models/checkpoints/yolov8n_fridge_v1/`

### Starting Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---:|---|
| `epochs` | 50 | Long enough for an early learning trend without overcommitting compute |
| `imgsz` | 640 | YOLOv8 default balance of speed and accuracy |
| `batch` | 16 | First-pass GPU batch size; reduce to 8 if memory is tight |
| `patience` | 15 | Stops wasted epochs if validation does not improve |
| `lr0` | default | Keep Ultralytics default for the first baseline run |

### Launch Command

```powershell
python train_yolov8_fridge.py
```

For a readiness-only check:

```powershell
python train_yolov8_fridge.py --dry-run
```

If local CUDA is unavailable or too slow, run the same command in Colab after uploading or mounting the project data. The script prints CUDA availability before training starts.

### Model Size Decision Framework

Start with `yolov8n.pt` because it trains fastest and gives the quickest feedback on data quality. Upgrade to `yolov8s.pt` after the first validation review only if nano plateaus with low mAP or unstable losses.

## Day 9 - Monitoring Checklist

Review the run with:

```powershell
python monitor_yolov8_training.py --copy-best
```

Run interim validation from the current best checkpoint with:

```powershell
python monitor_yolov8_training.py --val --copy-best
```

### Findings Template

| Field | Value |
|---|---|
| Epoch reached | TBD |
| mAP@0.5 | TBD |
| mAP@0.5:0.95 | TBD |
| Train loss trend | TBD |
| Validation loss trend | TBD |
| Overfitting signal | TBD |
| Nano vs. small decision | TBD |
| Labeled checkpoint | `models/checkpoints/yolov8n_fridge_v1_day9_best.pt` |

### Decision Rule

| Observation | Action |
|---|---|
| mAP@0.5 trends toward `>0.5` by roughly epoch 30-40 and losses are healthy | Continue with `yolov8n.pt` |
| mAP@0.5 stays below roughly `0.3-0.4` with flat losses | Escalate to `yolov8s.pt` using the same starting hyperparameters |
| Training crashes repeatedly or reports label/class errors | Pause training and fix data quality before changing model size |

## Day 10 Handoff Notes

Day 10 should use the final `best.pt` for full validation against the held-out split and compare against the MVP target of mAP@0.5 greater than `0.6`.

## Day 11 - Inference Module Added

Implemented a reusable inference wrapper in `backend/modules/detect_ingredients.py`.

### What is included

- `detect_ingredients(image_path, conf_threshold=0.3)` entry point
- Model loaded once at module import time
- Basic handling for missing files and empty detections
- Unit tests in `tests/test_detect_ingredients.py`

### Verification status

The Day 11 tests were run inside a workspace-local virtual environment on the D drive:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_detect_ingredients.py
```

Result: `1 passed, 2 skipped`.
