# Plate Project — Python Files Reference

## File Map (by importance to you)

```
plate_project/
├── detect_rec.py      ★ Main script — detection + OCR + authorization
├── debug_ocr.py       ★ Debug helper — visualize OCR preprocessing
├── export.py          ▸ Convert model to other formats (ONNX, TFLite, etc.)
├── val.py             ▸ Evaluate model accuracy on a test dataset
├── models/
│   ├── common.py      ▸ All YOLO layer building blocks
│   ├── yolo.py        ▸ YOLO model assembly & forward pass
│   ├── experimental.py▸ Experimental/extra model layers
│   └── tf.py          ▸ TensorFlow/Keras version of the model
└── utils/
    ├── general.py     ▸ Core utilities (NMS, logging, file helpers)
    ├── dataloaders.py ▸ Image/video/webcam data feeding
    ├── torch_utils.py ▸ PyTorch device/model utilities
    ├── plots.py       ▸ Drawing bounding boxes, saving images
    ├── augmentations.py ▸ Image augmentation (training)
    ├── metrics.py     ▸ mAP, precision, recall calculation
    ├── loss*.py       ▸ Training loss functions
    ├── callbacks.py   ▸ Training event hooks
    ├── autoanchor.py  ▸ Anchor box auto-tuning
    ├── autobatch.py   ▸ Auto batch size finder
    ├── downloads.py   ▸ Model/dataset download helpers
    ├── activations.py ▸ Custom activation functions
    ├── lion.py        ▸ Lion optimizer
    ├── triton.py      ▸ Triton inference server client
    └── loggers/       ▸ WandB / ClearML / Comet training loggers
```

---

## ★ `detect_rec.py` — The Main Script

**Purpose:** Runs the full pipeline: YOLO detects plates → PaddleOCR reads Arabic text → checks authorization against the server database.

### Global Constants & Config
| Name | What it does |
|---|---|
| `SERVER_URL` | Smart Campus backend URL |
| `OCR_EXCLUDE_WORDS` | Set of EGYPT/مصر watermark words to ignore |
| `COLOR_*` | BGR colors for drawing bounding boxes |
| `MIN_READINGS_FOR_CONSENSUS` | How many OCR readings needed before committing to a result (video mode) |

---

### Free Functions

| Function | What it does |
|---|---|
| `calculate_iou(box1, box2)` | Calculates Intersection-over-Union between two bounding boxes. Used by the tracker to match the same plate across frames. |
| `fix_arabic_text(text)` | Reshapes Arabic text for correct RTL display using `arabic_reshaper` + `python-bidi`. |
| `is_video_source(source_path)` | Returns `True` if the source is a webcam number, video file, or stream URL. |
| `normalize_image(img)` | Converts any image format (RGBA, grayscale, float32, uint16) to standard BGR uint8 so OpenCV and PaddleOCR can work with it. |

---

### Class `PlateTracker`

> Used only in **video/webcam mode**. Tracks each plate across multiple frames and builds a consensus reading from repeated OCR results.

| Method | What it does |
|---|---|
| `__init__()` | Sets up tracking dictionaries and thresholds. |
| `update(bbox, ocr_text, ...)` | Called every frame. Assigns a track ID to each detected plate, accumulates OCR readings, and finalizes when enough consistent readings are collected. |
| `_find_matching_plate(bbox)` | Internal — finds an existing tracked plate whose bounding box overlaps this one (using IoU). |
| `_calculate_consensus(readings)` | Internal — picks the most frequent OCR reading and returns its confidence score. |
| `should_perform_ocr(track_id)` | Returns `False` once a plate is finalized, preventing wasted OCR calls. |
| `get_tracking_info(track_id)` | Returns debug info (readings count, frames tracked, status) for a given plate. |
| `cleanup_old_tracks(current_bboxes)` | Deletes finalized tracks that have been tracked for over 100 frames. |

---

### Class `ArabicPlateRecognizer`

> The OCR + authorization engine. Holds the PaddleOCR instance and the plate database.

| Method | What it does |
|---|---|
| `__init__(device, db_path)` | Initializes PaddleOCR and loads the plate database. |
| `_load_database(db_path)` | Loads authorized plates. Priority: **1** → live server fetch, **2** → local JSON file, **3** → hardcoded fallback. |
| `_enhance_plate(img)` *(static)* | Generates 5 preprocessed versions of the plate crop: `original`, `clahe`, `clahe+sharp`, `otsu`, `otsu_inv`. OCR is tried on each variant. |
| `_count_arabic_chars(text)` *(static)* | Counts non-ASCII, non-space characters (Arabic letters + Arabic-Indic digits). Used to score OCR results. |
| `recognize_plate(img, bbox)` | **Main OCR entry point.** Crops the plate, upscales to ≥160×400 px, runs all 5 preprocessing variants through PaddleOCR, and returns the best result (highest confidence-weighted Arabic character score). Filters out English letters, Western digits, and EGYPT watermarks. |
| *(inner)* `_space_chars_rtl(text)` | Reverses characters and adds spaces between them for correct Arabic RTL display (e.g. `صون٩٤٣٢` → `ص و ن ٩ ٤ ٣ ٢`). Strips any ASCII characters. |
| *(inner)* `_ocr_variant(variant_img)` | Runs PaddleOCR on one preprocessed image, applies all filters (confidence gate, watermark exclusion, ASCII exclusion), and returns `(regions, text, weighted_score)`. |
| `normalize_plate(text)` *(static)* | Strips spaces and converts Western digits (0-9) to Arabic-Indic (٠-٩). Makes manually entered plates match OCR output for database comparison. |
| `check_security(plate_text)` | Normalizes the OCR result and compares it against every entry in the database. Returns `(status, is_authorized, in_database)`. |

---

### `run(...)` — The Main Detection Loop

**Entry point for all three modes:**

| Mode | How it's triggered |
|---|---|
| **Webcam** | `--source 0` (or any number) |
| **Image** | `--source image.jpg` |
| **Video** | `--source video.mp4` |

**Webcam mode** uses two threads:
- **Inference thread** (`inference_worker`): always grabs the latest frame, runs YOLO + OCR, updates overlays.
- **Display loop**: reads camera at full FPS, draws last-known detections.

**Image/Video mode**: standard YOLO loop — load → infer → NMS → OCR → save results.

| Helper function inside `run` | What it does |
|---|---|
| `inference_worker()` | Webcam-only thread: YOLO → NMS → OCR → log results |

---

### `parse_opt()` and `main(opt)`
| Function | What it does |
|---|---|
| `parse_opt()` | Parses all command-line arguments (`--weights`, `--source`, `--conf-thres`, `--db-path`, etc.) |
| `main(opt)` | Validates requirements and calls `run(**vars(opt))` |

---

## ★ `debug_ocr.py` — OCR Debugging Tool

**Purpose:** A one-shot script that loads the last saved plate crop and runs PaddleOCR on all preprocessing variants, printing results and saving debug images.

**How to use:**
```powershell
# 1. Run detection with --save-crop to save the plate image:
.\venv312\Scripts\python detect_rec.py --weights best.pt --source .\test.jpeg --save-crop

# 2. Copy the crop to a safe ASCII filename:
.\venv312\Scripts\python -c "import shutil,glob; shutil.copy(sorted(glob.glob('runs/detect/exp*/crops/licence/*.jpg'))[-1],'debug_plate_crop.jpg')"

# 3. Run the debug script:
.\venv312\Scripts\python debug_ocr.py
```

**Output:** Prints each variant's raw OCR tokens + confidence, and saves `debug_<variant>.png` images to inspect.

---

## ▸ `export.py` — Model Format Exporter

**Purpose:** Converts the trained `best.pt` PyTorch model to other deployment formats.

| Function | Exports to |
|---|---|
| `export_formats()` | Lists all supported formats |
| `export_torchscript()` | TorchScript (`.torchscript`) |
| `export_onnx()` | ONNX (`.onnx`) — most portable |
| `export_openvino()` | Intel OpenVINO |
| `export_paddle()` | PaddlePaddle |
| `export_coreml()` | Apple CoreML (iOS/macOS) |
| `export_engine()` | TensorRT (NVIDIA GPU) |
| `export_saved_model()` | TensorFlow SavedModel |
| `export_pb()` | TensorFlow GraphDef (`.pb`) |
| `export_tflite()` | TensorFlow Lite (mobile) |
| `export_edgetpu()` | Google Coral Edge TPU |
| `export_tfjs()` | TensorFlow.js (browser) |
| `add_tflite_metadata()` | Embeds metadata into TFLite file |
| `run()` | Orchestrates the export based on chosen format |
| `parse_opt()` | Parses `--include`, `--weights`, `--imgsz`, etc. |
| `main(opt)` | Entry point |

**You would use this if** you want to deploy `best.pt` to a mobile app (TFLite), a browser (TF.js), or a faster GPU engine (TensorRT).

---

## ▸ `val.py` — Model Validation

**Purpose:** Measures how accurate your trained model is on a test image dataset. Reports mAP (mean Average Precision).

| Function | What it does |
|---|---|
| `save_one_txt(predn, ...)` | Saves one image's detections as a YOLO `.txt` label file |
| `save_one_json(predn, ...)` | Saves detections in COCO JSON format |
| `process_batch(detections, labels, iouv)` | Computes True Positives for a batch of predictions vs ground-truth labels |
| `run(...)` | Full validation loop: loads dataset → runs model → computes mAP, precision, recall |
| `parse_opt()` | Parses `--weights`, `--data`, `--conf-thres`, etc. |
| `main(opt)` | Entry point |

**You would use this to** check if your model is still accurate after retraining: `python val.py --weights best.pt --data data/coco128.yaml`

---

## ▸ `models/` — YOLO Model Architecture

| File | What it does |
|---|---|
| `common.py` | All individual layer types: `Conv`, `C2f`, `SPPF`, `Concat`, `Detect`, etc. Building blocks of the network. |
| `yolo.py` | Assembles layers into a full `DetectionModel`. Handles `.yaml` config parsing, forward pass, and NMS post-processing. |
| `experimental.py` | Extra/experimental layers like `CrossConv`, `MixConv2d`, `Ensemble` (combining multiple models). |
| `tf.py` | TensorFlow/Keras reimplementation of the same model layers, used by `export.py` when exporting to TF formats. |

---

## ▸ `utils/` — Support Library

| File | What it does |
|---|---|
| `general.py` | The most-used utility file. Contains NMS (`non_max_suppression`), box scaling (`scale_boxes`), path helpers, logging setup, `check_img_size`, etc. |
| `dataloaders.py` | `LoadImages`, `LoadStreams`, `LoadScreenshots` — feeds frames into the detection pipeline from any source. |
| `torch_utils.py` | `select_device` (CPU/GPU), model profiling, smart inference mode decorator. |
| `plots.py` | `Annotator` class — draws boxes and labels on images. `save_one_box` — crops and saves a detected region. |
| `augmentations.py` | Training-time image augmentation: mosaic, random flip, HSV shift, letterbox resize. |
| `metrics.py` | `ap_per_class`, `box_iou`, confusion matrix — used by `val.py`. |
| `loss.py` / `loss_tal*.py` | Different training loss functions (BCE + CIoU). `loss_tal` = Task-Aligned Learning variant. |
| `callbacks.py` | `Callbacks` class — hooks for training events (on_train_start, on_epoch_end, etc.). |
| `autoanchor.py` | Automatically tunes anchor box sizes to match your dataset. |
| `autobatch.py` | Finds the largest batch size that fits in GPU memory. |
| `downloads.py` | Downloads pretrained weights and datasets from URLs. |
| `activations.py` | Custom activations: `SiLU`, `Mish`, `FReLU`, etc. |
| `lion.py` | Lion optimizer (alternative to Adam/SGD). |
| `triton.py` | Client for NVIDIA Triton inference server (production deployment). |
| `loggers/` | Integrations with WandB, ClearML, and Comet for training experiment tracking. |
