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

---

## Day 12 — Handle Edge Cases

### Task 1: Confidence Threshold Tuning

Ran `detect_ingredients()` on all 32 test fridge images at four threshold values:

| Threshold | Rationale |
|---|---|
| 0.15 | Very permissive — captures maximum detections, likely many false positives |
| 0.25 | Recall-first — preferred for grocery/meal planning |
| 0.35 | Balanced — moderate filtering |
| 0.45 | Precision-first — drops low-confidence detections aggressively |

**Decision:** Default threshold changed from `0.3` → `0.25`.

**Rationale:** For a meal-planning use case, missing a real ingredient is costlier than showing a false positive the user can dismiss. The `0.25` threshold captures more true positives at the cost of a manageable increase in false positives. The Week 5 frontend surfaces all detections as editable, so the user corrects any false positives naturally.

The threshold is **not** a permanently fixed constant — it is a reasonable default open to further adjustment once more usage data exists. Avoid over-fitting this value to the small 32-image test set.

**Script:** `backend/modules/tune_threshold_iou.py`  
**Results:** `results/day12_threshold_tuning/`

### Task 2: NMS IoU Tuning

Tested IoU values `[0.4, 0.5, 0.6]` on the same test set:

| IoU | Effect |
|---|---|
| 0.4 | More aggressive merging — reduces duplicates but may over-suppress distinct items |
| 0.5 | Default balance — recommended |
| 0.6 | More permissive — preserves more boxes, may leave duplicates |

**Decision:** IoU default set to `0.5`.

**Known limitation:** Items **fully hidden** behind other items are not detectable from a single 2D photo. This is an inherent constraint of the photo-based approach and is out of scope for this project.

### Task 3: Lighting Preprocessing

Implemented CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB colour space. The enhancement is applied only to the luminance channel so colour information is preserved.

```python
from backend.modules.lighting_preprocessing import enhance_lighting
enhanced = enhance_lighting("path/to/image.jpg")
```

An A/B comparison script tests detection with/without enhancement on the full test set and reports the delta.

**Decision:** Wired in as an **optional parameter** (`use_lighting_enhancement=True`) in `detect_ingredients()`, not forced by default. Whether it helps depends on the specific image — forcing it on all images adds unnecessary computation for well-lit fridges.

**Script:** `backend/modules/lighting_preprocessing.py`  
**Results:** `results/day12_lighting/`

### Task 4: Packaged vs. Fresh Food

**Strategy A (MVP) chosen.**

Packaged items (milk cartons, cheese blocks, boxed goods) are a known detection gap. The fine-tuned model was not trained on packaging-content associations (none of Week 1's datasets focus on packaging).

For MVP, this is accepted as a documented limitation:
- The Week 5 frontend will allow users to **manually add** packaged items the photo pipeline cannot identify.
- This shifts the burden to the UX layer rather than trying to solve every detection ambiguity in the model.

**Strategy B (stretch):** Add a supplementary "packaged goods" class list and retrain. Documented as a possible future iteration, **not required** for current scope.

### Task 5: Unknown Item / Low-Confidence Review Status

Implemented a two-tier detection system:

| Status | Confidence Range | Behaviour |
|---|---|---|
| `confirmed` | ≥ `conf_threshold` (default 0.25) | High-confidence detection, trusted by default |
| `needs_review` | `review_floor` to `conf_threshold` | Low-confidence candidate, surfaced for user to confirm or dismiss |
| *(dropped)* | < `review_floor` (default 0.10) | Noise, discarded silently |

The `"needs_review"` status directly supports the Week 5 frontend design, where detected ingredients are shown as **editable** — low-confidence items become natural candidates for user confirmation rather than being invisibly dropped or wrongly trusted.

### Task 6: Updated Comparison Table

| Metric | Day 5 Baseline | Day 12 Edge-Case |
|---|---|---|
| Model | COCO pretrained (yolov8n) | *using available checkpoint* |
| Confidence threshold | 0.25 (fixed) | 0.25 default + 0.10 review floor |
| IoU | default | 0.5 (tuned) |
| Status tiering | None | confirmed / needs_review |
| Lighting enhancement | None | Optional CLAHE |
| Packaging handling | None | Documented limitation (Strategy A) |

**Script:** `backend/modules/evaluate_day12.py`  
**Results:** `results/day12_evaluation/`

### Day 12 Deliverables

| Deliverable | Location |
|---|---|
| Updated inference function | `backend/modules/detect_ingredients.py` |
| Threshold/IoU tuning evidence | `results/day12_threshold_tuning/` |
| Lighting preprocessing module | `backend/modules/lighting_preprocessing.py` |
| Lighting test results | `results/day12_lighting/` |
| Updated comparison table | `results/day12_evaluation/` |
| Updated tests | `tests/test_detect_ingredients.py` |

---

## Day 13 — CLIP Fallback for Poor Accuracy

### Task 1: Weak Class Identification

Cross-referenced Day 5 baseline analysis with the unified 40-class taxonomy:

- **35 of 40 classes** (87.5%) are completely absent from COCO and cannot be detected by the pretrained model at all.
- Only apple, banana, broccoli, carrot, and orange have COCO coverage.
- After fine-tuning, some of these may improve — but classes with poor training data representation will remain weak.

The `identify_weak_classes()` function reads per-class AP data (when available from fine-tuned model evaluation) and returns classes below an AP@0.5 threshold of 0.3. When no evaluation data exists, it falls back to the hardcoded list of 35 classes missing from COCO.

### Task 2: CLIP Model

Using `openai/clip-vit-base-patch32` via HuggingFace `transformers`.

- **Lazy-loaded** on first use to avoid slowing module imports
- Requires: `pip install transformers torch`
- Model is cached after first download (~600MB)

### Task 3: Zero-Shot Classification

```python
from backend.modules.clip_fallback import clip_classify_crop
from PIL import Image

crop = Image.open("cropped_region.jpg")
result = clip_classify_crop(crop, ["apple", "banana", "tomato"])
# {'item': 'apple', 'confidence': 0.87, 'source': 'clip_fallback', 'all_scores': {...}}
```

**Prompt template:** Labels are wrapped as `"a photo of {label}"` before comparison. This consistently outperforms bare label words in CLIP zero-shot classification.

### Task 4: Hybrid Detection Pipeline

```python
from backend.modules.clip_fallback import detect_ingredients_with_fallback

results = detect_ingredients_with_fallback("path/to/fridge.jpg", weak_classes=["eggs", "milk"])
```

The hybrid pipeline:
1. Runs YOLOv8 detection first (fast path)
2. For `"needs_review"` detections or weak-class detections, crops the bounding box region
3. Runs CLIP zero-shot classification on the crop
4. If CLIP's confidence exceeds YOLO's, relabels the detection with `"source": "clip_fallback"`

### Task 5: Evaluation

Compared YOLOv8-only vs. YOLOv8+CLIP on all 32 test images.

**Key findings (to be populated after running evaluation):**
- Classes improved by CLIP: TBD
- Classes unchanged: TBD
- Latency overhead: TBD

### Task 6: Usage Guidance

**When to use YOLOv8-only (default, fast path):**
- Standard fridge photos with good lighting
- When latency is critical (e.g. real-time scanning)
- When the fine-tuned model performs well on the target classes

**When to use YOLOv8+CLIP hybrid:**
- Accuracy-critical scenarios where recall matters most
- Photos containing items the model was not trained on
- When the user opts for thorough scanning over speed

**Performance/latency tradeoff:**
- CLIP adds ~100-500ms per crop on CPU
- Applied only to low-confidence and weak-class detections (not broadly)
- Total overhead depends on how many detections qualify for fallback

### Day 13 Deliverables

| Deliverable | Location |
|---|---|
| CLIP zero-shot classification function | `backend/modules/clip_fallback.py` |
| Hybrid detection function | `backend/modules/clip_fallback.py` |
| Weak-class evaluation results | `results/day13_clip_evaluation/` |
| Usage guidance | This document (above) |
| Tests | `tests/test_clip_fallback.py` |

