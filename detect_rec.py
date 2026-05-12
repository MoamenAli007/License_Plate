import argparse
import json
import os
import re

# Disable PaddlePaddle OneDNN — avoids NotImplementedError on Windows with Paddle 3.x
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import platform
import sys

# Force UTF-8 output on Windows so Arabic text in logs doesn't crash cp1252 terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Tuple, Optional
from collections import Counter
from datetime import datetime
import requests

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Import PaddleOCR for Arabic
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("ERROR: PaddleOCR not installed. Run: pip install paddleocr==2.7.3")
    sys.exit(1)

# Import Arabic RTL support
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    RTL_SUPPORT = True
except ImportError:
    print("WARNING: Install for proper Arabic display: pip install arabic-reshaper python-bidi")
    RTL_SUPPORT = False

# Constants
PIXEL_NORMALIZATION = 255
MS_CONVERSION = 1E3

# ── Smart Campus Server ───────────────────────────────────────
# Set this to your server's address (same machine = localhost).
# If running on a different PC/server, use its IP, e.g.:
#   SERVER_URL = "http://192.168.1.10:8000"
SERVER_URL = "http://smartcampus.engineer"
SERVER_TIMEOUT = 5  # seconds to wait for a response
# ─────────────────────────────────────────────────────────────
DEFAULT_FPS = 30

# Tracking constants
MIN_READINGS_FOR_CONSENSUS = 3
CONFIDENCE_THRESHOLD = 0.6
MAX_TRACKING_FRAMES = 30
IOU_THRESHOLD_TRACKING = 0.3

# Color constants (BGR format)
COLOR_AUTHORIZED = (0, 255, 0)  # Green
COLOR_DENIED = (0, 0, 255)  # Red
COLOR_UNKNOWN = (255, 0, 0)  # Blue
COLOR_TRACKING = (0, 255, 255)  # Yellow

# ── OCR word exclusion ────────────────────────────────────────
# Words printed on every Egyptian plate that are NOT part of the
# plate number itself. Comparison is case-insensitive.
OCR_EXCLUDE_WORDS = {
    # Latin variants of EGYPT
    'egypt', 'egyp', 'egypr', 'egpt', 'egvpt', 'e g y p t',
    'egt', 'egyt', 'egyp', 'pte', 'pt', 'epy',
    'eg', 'gy', 'gypt',
    # Arabic variants
    'مصر', 'مصري',
    'رصم', 'ص م ر', 'مصر.',
}
# Any token containing 'egy' (catches EGVPT, EGYPT, EGYP etc.)
OCR_EXCLUDE_CONTAINS = {'egy', 'egypt'}
# ─────────────────────────────────────────────────────────────


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def fix_arabic_text(text):
    """Fix Arabic text for proper RTL display"""
    if not RTL_SUPPORT or not text:
        return text
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        rtl_text = get_display(reshaped_text)
        return rtl_text
    except Exception as e:
        LOGGER.warning(f"RTL conversion failed: {e}")
        return text


def is_video_source(source_path):
    """Check if source is a video file or stream"""
    source = str(source_path)

    # Check if it's a webcam/stream
    if source.isnumeric():
        return True
    if source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        return True

    # Check file extension
    if Path(source).suffix[1:].lower() in VID_FORMATS:
        return True

    return False


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to ensure consistent format across different file types.
    Handles: PNG with alpha, grayscale, different bit depths, etc.
    """
    if img is None or img.size == 0:
        return img
    
    # Convert to uint8 if not already
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Handle different channel configurations
    if len(img.shape) == 2:
        # Grayscale - convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # RGBA/BGRA - remove alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:
            # Single channel - convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            # Already BGR or RGB - ensure it's BGR
            # Most OpenCV operations expect BGR
            pass
    
    return img


class PlateTracker:
    """Tracks license plates across video frames"""

    def __init__(self, min_readings=MIN_READINGS_FOR_CONSENSUS,
                 confidence_threshold=CONFIDENCE_THRESHOLD,
                 max_frames=MAX_TRACKING_FRAMES):
        self.plates = {}
        self.next_id = 0
        self.min_readings = min_readings
        self.confidence_threshold = confidence_threshold
        self.max_frames = max_frames

    def update(self, bbox, ocr_text, security_status=None, is_in_database=False):
        """Update tracker with new detection"""
        track_id = self._find_matching_plate(bbox)

        if track_id is None:
            track_id = self.next_id
            self.next_id += 1
            self.plates[track_id] = {
                'bbox': bbox,
                'readings': [],
                'frames_tracked': 0,
                'finalized': False,
                'consensus_text': None,
                'confidence': 0.0,
                'security_status': None,
                'is_authorized': None,
                'stop_ocr': False
            }

        plate_data = self.plates[track_id]
        plate_data['bbox'] = bbox
        plate_data['frames_tracked'] += 1

        should_continue_ocr = not plate_data['stop_ocr']

        # Add valid readings
        if should_continue_ocr and ocr_text and ocr_text not in ['CROP_ERROR', 'NO_TEXT', 'OCR_ERROR']:
            plate_data['readings'].append(ocr_text)
            LOGGER.debug(f"Track {track_id}: Added '{ocr_text}' (total: {len(plate_data['readings'])})")

        # Immediate stop if database match
        if is_in_database and ocr_text and not plate_data['finalized']:
            plate_data['consensus_text'] = ocr_text
            plate_data['confidence'] = 1.0
            plate_data['finalized'] = True
            plate_data['security_status'] = security_status[0]
            plate_data['is_authorized'] = security_status[1]
            plate_data['stop_ocr'] = True
            LOGGER.info(f"⚡ DATABASE MATCH: {ocr_text} - {security_status[0]}")
            return (track_id, plate_data['consensus_text'], plate_data['confidence'],
                    plate_data['finalized'], (plate_data['security_status'], plate_data['is_authorized']),
                    should_continue_ocr)

        # Normal consensus logic
        if not plate_data['finalized']:
            if (len(plate_data['readings']) >= self.min_readings or
                    plate_data['frames_tracked'] >= self.max_frames):
                plate_data['consensus_text'], plate_data['confidence'] = self._calculate_consensus(
                    plate_data['readings']
                )

                if plate_data['confidence'] >= self.confidence_threshold:
                    plate_data['finalized'] = True
                    if security_status:
                        plate_data['security_status'] = security_status[0]
                        plate_data['is_authorized'] = security_status[1]
                    plate_data['stop_ocr'] = True
                    LOGGER.info(f"✓ CONSENSUS: {plate_data['consensus_text']} " +
                                f"({plate_data['confidence']:.0%} from {len(plate_data['readings'])} readings)")

        return (track_id,
                plate_data['consensus_text'],
                plate_data['confidence'],
                plate_data['finalized'],
                (plate_data['security_status'], plate_data['is_authorized']),
                should_continue_ocr)

    def _find_matching_plate(self, bbox):
        """Find existing tracked plate matching this bbox"""
        best_iou = IOU_THRESHOLD_TRACKING
        best_id = None

        for track_id, plate_data in self.plates.items():
            iou = calculate_iou(bbox, plate_data['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        return best_id

    def _calculate_consensus(self, readings):
        """Calculate consensus from multiple readings"""
        if not readings:
            return None, 0.0

        counter = Counter(readings)
        most_common_text, most_common_count = counter.most_common(1)[0]
        confidence = most_common_count / len(readings)

        LOGGER.debug(f"Consensus: {most_common_text} ({confidence:.0%}) from {counter.most_common(3)}")
        return most_common_text, confidence

    def should_perform_ocr(self, track_id):
        """Check if OCR should be performed"""
        if track_id in self.plates:
            return not self.plates[track_id]['stop_ocr']
        return True

    def get_tracking_info(self, track_id):
        """Get tracking info for a plate"""
        if track_id in self.plates:
            plate_data = self.plates[track_id]
            return {
                'readings_count': len(plate_data['readings']),
                'frames_tracked': plate_data['frames_tracked'],
                'all_readings': plate_data['readings'],
                'finalized': plate_data['finalized'],
                'stop_ocr': plate_data['stop_ocr'],
                'security_status': plate_data['security_status'],
                'is_authorized': plate_data['is_authorized']
            }

        return None

    def cleanup_old_tracks(self, current_bboxes):
        """Remove old tracks no longer visible"""
        to_remove = []
        for track_id in list(self.plates.keys()):
            if self.plates[track_id]['finalized'] and self.plates[track_id]['frames_tracked'] > 100:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.plates[track_id]


class ArabicPlateRecognizer:
    """PaddleOCR-based Arabic license plate recognizer with JSON database"""

    def __init__(self, device='cpu', db_path: Optional[Path] = None):
        try:
            use_gpu = device != 'cpu' and torch.cuda.is_available()
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=False, show_log=False)
            LOGGER.info(f"PaddleOCR initialized (Arabic, CPU mode)")
        except Exception as e:
            LOGGER.error(f"Failed to initialize PaddleOCR: {e}")
            raise

        self.security_db = self._load_database(db_path)

    def _load_database(self, db_path: Optional[Path]) -> dict:
        """Load security database.

        Priority order:
        1. Live fetch from the Smart Campus FastAPI server (GET /iot/license-plates)
        2. Local JSON file (if --db-path is provided and the file exists)
        3. Hardcoded default examples (offline fallback)
        """
        # ── 1. Try the live server first ──────────────────────────
        try:
            url = f"{SERVER_URL}/iot/license-plates"
            response = requests.get(url, timeout=SERVER_TIMEOUT)
            response.raise_for_status()
            plates = response.json()  # list of {id, plate_number, owner_name, is_allowed}

            db = {}
            for entry in plates:
                plate_number = entry.get("plate_number", "").strip()
                is_allowed   = entry.get("is_allowed", False)
                if plate_number:
                    db[plate_number] = "AUTHORIZED" if is_allowed else "DENIED - BLACKLISTED"

            LOGGER.info(f"✓ Loaded {len(db)} plates from server ({url})")
            return db

        except requests.exceptions.ConnectionError:
            LOGGER.warning(f"⚠ Server not reachable at {SERVER_URL} — falling back to local database")
        except requests.exceptions.Timeout:
            LOGGER.warning(f"⚠ Server timed out ({SERVER_TIMEOUT}s) — falling back to local database")
        except Exception as e:
            LOGGER.warning(f"⚠ Server fetch failed: {e} — falling back to local database")

        # ── 2. Try the local JSON file ────────────────────────────
        if db_path and Path(db_path).exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    db = json.load(f)
                LOGGER.info(f"✓ Loaded {len(db)} plates from local file {db_path}")
                return db
            except Exception as e:
                LOGGER.warning(f"Failed to load local database from {db_path}: {e}")

        # ── 3. Hardcoded fallback ─────────────────────────────────
        # Keys must match EXACTLY what the OCR returns (after RTL reversal).
        # Use plates_log.txt to find the exact string for any new plate.
        default_db = {
            # ── KK.jpg test plate ─────────────────────────────────
            "ص و ن 9 4 3 2": "AUTHORIZED",  
            "ر و ص 5 6 1": "AUTHORIZED",
              
        }
        LOGGER.info(f"Using hardcoded default database ({len(default_db)} plates)")
        return default_db

    def recognize_plate(self, img: np.ndarray, bbox: list) -> str:
        """
        Recognize Arabic text from license plate.
        - Upscales the crop so PaddleOCR can read small Arabic characters.
        - Strips words that appear on every Egyptian plate (EGYPT / مصر).
        - Sorts detected text regions right-to-left for correct Arabic order.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Padding to avoid cutting edge characters
            pad = 4
            h, w = img.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            cropped = img[y1:y2, x1:x2]

            if cropped.size == 0 or cropped.shape[0] < 15 or cropped.shape[1] < 30:
                return "CROP_ERROR"

            # Normalize format (handles RGBA, grayscale, different bit-depths)
            cropped = normalize_image(cropped)
            if cropped is None or cropped.size == 0:
                return "CROP_ERROR"

            # ── Upscale to improve OCR on small plates ────────────────
            # PaddleOCR struggles with crops shorter than ~64 px.
            min_h = 80
            ch, cw = cropped.shape[:2]
            if ch < min_h:
                scale  = min_h / ch
                new_w  = int(cw * scale)
                cropped = cv2.resize(cropped, (new_w, min_h),
                                     interpolation=cv2.INTER_CUBIC)

            # ── OCR ───────────────────────────────────────────────────
            result = self.ocr.ocr(cropped, cls=True)

            if result is None or len(result) == 0 or result[0] is None:
                return "NO_TEXT"

            # ── Collect text regions, filter excluded words ───────────
            text_regions = []

            for line in result[0]:
                if not line:
                    continue
                bbox_ocr = line[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                text     = line[1][0]
                conf     = line[1][1]

                if not text or not text.strip():
                    continue

                # Skip words that are printed on every Egyptian plate
                if text.strip().lower() in OCR_EXCLUDE_WORDS:
                    LOGGER.debug(f"Excluded OCR token: '{text.strip()}'")
                    continue

                # Skip tokens that are entirely Latin/English letters (e.g. EGYPT, A, B)
                clean_token = re.sub(r'\s+', '', text.strip())
                if clean_token and re.fullmatch(r'[A-Za-z]+', clean_token):
                    LOGGER.debug(f"Excluded all-Latin OCR token: '{text.strip()}'")
                    continue

                # Use bounding-box center X for right-to-left sorting
                xs       = [pt[0] for pt in bbox_ocr]
                x_center = (min(xs) + max(xs)) / 2

                text_regions.append({'text': text.strip(), 'x': x_center, 'conf': conf})

            if not text_regions:
                return "NO_TEXT"

            # ── Sort RIGHT → LEFT, join ───────────────────────────────
            text_regions.sort(key=lambda r: r['x'], reverse=True)

            # Split every character with a space AND reverse for correct RTL order
            # PaddleOCR scans LTR physically, so ٢٣٤٩نو ص → reversed → ص و ن ٩ ٤ ٣ ٢
            def _space_chars_rtl(text: str) -> str:
                # Exclude individual English/Latin letters; keep Arabic chars and digits
                chars = [c for c in text if not c.isspace() and not c.isascii() or c.isdigit()]
                return ' '.join(reversed(chars))

            recognized_text = '  '.join(
                _space_chars_rtl(r['text']) for r in text_regions
            ).strip()

            LOGGER.info(f"OCR regions ({len(text_regions)}): "
                        + " | ".join(f"'{r['text']}' @x={r['x']:.0f}" for r in text_regions))
            LOGGER.info(f"Plate text: [{recognized_text}]")
            return recognized_text if recognized_text else "NO_TEXT"

        except Exception as e:
            LOGGER.error(f"OCR failed: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            return "OCR_ERROR"

    @staticmethod
    def normalize_plate(text: str) -> str:
        """
        Normalize a plate string so that manually typed entries (from the mobile
        app) always match what the OCR produces, regardless of:
          - spaces between characters  (ص و ن  ==  صون)
          - Western vs Arabic-Indic digits  (2349  ==  ٢٣٤٩)
          - trailing/leading whitespace
        """
        # Remove ALL spaces
        text = text.replace(' ', '')
        # Convert Western digits 0-9 → Arabic-Indic ٠-٩
        western_to_arabic = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
        text = text.translate(western_to_arabic)
        return text.strip()

    def check_security(self, plate_text: str) -> Tuple[str, bool, bool]:
        """
        Check plate against the database.
        Normalizes both sides so spacing and digit style don't cause mismatches.
        """
        normalized_ocr = self.normalize_plate(plate_text)

        for db_key, db_status in self.security_db.items():
            if self.normalize_plate(db_key) == normalized_ocr:
                is_authorized = db_status == "AUTHORIZED"
                return db_status, is_authorized, True

        return "UNKNOWN - ACCESS DENIED", False, False


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
        db_path=None,
        min_readings=MIN_READINGS_FOR_CONSENSUS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_url and is_file:
        source = check_file(source)

    # Detect if source is video or image
    is_video = is_video_source(source)
    if is_video:
        LOGGER.info(f"?? VIDEO MODE: Using frame tracking with consensus (min {min_readings} readings)")
    else:
        LOGGER.info(f"?? IMAGE MODE: Single OCR reading per plate")

    # Setup directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    if save_crop:
        (save_dir / 'crops').mkdir(parents=True, exist_ok=True)

    # Initialize components
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Initialize recognizer
    recognizer = ArabicPlateRecognizer(device=device.type, db_path=db_path)

    # Initialize tracker ONLY for video
    tracker = PlateTracker(min_readings=min_readings, confidence_threshold=confidence_threshold) if is_video else None

    # ──────────────────────────────────────────────────────────────────
    # WEBCAM FAST MODE: threaded inference so display runs at camera FPS
    # ──────────────────────────────────────────────────────────────────
    if webcam:
        import threading, time as _time

        view_img = check_imshow(warn=True)
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            LOGGER.error(f"Cannot open webcam {source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ── Shared state ───────────────────────────────────────────────
        # Use a single "latest frame" instead of a queue so inference
        # always acts on the NEWEST frame, never a stale buffered one.
        latest_frame      = [None]          # [0] = newest raw frame
        frame_lock        = threading.Lock()
        result_lock       = threading.Lock()
        last_overlay      = []              # (xyxy_int, label, color)
        stop_event        = threading.Event()

        # OCR cooldown: don't re-OCR the same region for OCR_COOLDOWN seconds
        OCR_COOLDOWN      = 5.0
        last_ocr_time     = {}             # bbox_key → timestamp

        # Reduce resolution for webcam to speed up YOLO (~500ms vs 1800ms)
        webcam_sz         = [320, 320]
        webcam_sz         = check_img_size(webcam_sz, s=stride)

        LOGGER.info("WEBCAM LIVE MODE: always processes latest frame, OCR cooldown 3s")
        LOGGER.info("Press Q in the video window to quit.")

        # ── Inference thread ───────────────────────────────────────────
        def inference_worker():
            model.warmup(imgsz=(1, 3, *webcam_sz))
            while not stop_event.is_set():
                # Always grab the absolute latest frame
                with frame_lock:
                    frame = latest_frame[0]
                if frame is None:
                    _time.sleep(0.02)
                    continue

                # Preprocess at reduced resolution
                im  = cv2.resize(frame, (webcam_sz[1], webcam_sz[0]))
                im  = im[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→CHW
                im  = np.ascontiguousarray(im)
                im_t = torch.from_numpy(im).to(device).float() / 255.0
                if len(im_t.shape) == 3:
                    im_t = im_t[None]

                # YOLO inference
                pred = model(im_t)
                pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres,
                                           classes, agnostic_nms, max_det=max_det)

                overlays = []
                now = _time.time()
                for det in pred:
                    if len(det):
                        det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf_val, cls in reversed(det):
                            bbox     = [x.item() for x in xyxy]
                            xyxy_int = [int(x.item() if hasattr(x, 'item') else x) for x in xyxy]

                            # Cooldown key: bucket bbox to 80px grid to absorb plate movement
                            bbox_key = tuple(v // 80 for v in xyxy_int)
                            last_t   = last_ocr_time.get(bbox_key, 0)

                            if now - last_t >= OCR_COOLDOWN:
                                # Time to do OCR on this plate region
                                last_ocr_time[bbox_key] = now
                                ocr = recognizer.recognize_plate(frame, bbox)
                            else:
                                # Reuse last overlay entry if available, skip OCR
                                existing = next(
                                    (o for o in last_overlay if o[0] == xyxy_int), None)
                                if existing:
                                    overlays.append(existing)
                                else:
                                    overlays.append((xyxy_int, 'Scanning...', COLOR_TRACKING))
                                continue

                            if ocr and ocr not in ['CROP_ERROR', 'NO_TEXT', 'OCR_ERROR']:
                                status, is_auth, in_db = recognizer.check_security(ocr)
                                label = f'{ocr} | {status}'
                                color = COLOR_AUTHORIZED if is_auth else COLOR_DENIED

                                from datetime import datetime as _dt
                                _ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
                                with open(Path('plates_log.txt'), 'a', encoding='utf-8') as _lf:
                                    _lf.write(f"[{_ts}] plate='{ocr}'  status={status}  source=webcam\n")

                                run_result_path = save_dir / 'results.txt'
                                with open(run_result_path, 'a', encoding='utf-8') as _rf:
                                    _rf.write(
                                        f"[{_ts}]\n"
                                        f"  source   : webcam\n"
                                        f"  plate    : {ocr}\n"
                                        f"  escaped  : {ocr.encode('unicode_escape').decode()}\n"
                                        f"  status   : {status}\n"
                                        f"{'-'*40}\n"
                                    )

                                sep = '=' * 52
                                LOGGER.info(sep)
                                LOGGER.info(f"  PLATE DETECTED : {ocr}")
                                LOGGER.info(f"  STATUS         : {status}")
                                LOGGER.info(f"  Escaped        : {ocr.encode('unicode_escape').decode()}")
                                LOGGER.info(sep)
                            else:
                                label = 'Scanning...'
                                color = COLOR_TRACKING

                            overlays.append((xyxy_int, label, color))

                with result_lock:
                    last_overlay.clear()
                    last_overlay.extend(overlays)

        worker = threading.Thread(target=inference_worker, daemon=True)
        worker.start()

        # ── Display loop — runs at full camera FPS ─────────────────────
        win_name = 'License Plate Detection  [Q to quit]'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Always overwrite latest_frame with the newest capture
            with frame_lock:
                latest_frame[0] = frame.copy()

            # Draw last known detections on the live frame
            display = frame.copy()
            with result_lock:
                for xyxy_int, label, color in last_overlay:
                    x1, y1, x2, y2 = xyxy_int
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(display, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow(win_name, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        stop_event.set()
        worker.join(timeout=3)
        cap.release()
        cv2.destroyAllWindows()
        LOGGER.info("Webcam session ended.")
        return   # ← skip the rest of the function (image/video path)

    # ──────────────────────────────────────────────────────────────────
    # IMAGE / VIDEO FILE MODE  (unchanged below)
    # ──────────────────────────────────────────────────────────────────

    # Dataloader
    bs = 1
    if screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # Main detection loop
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= PIXEL_NORMALIZATION
            if len(im.shape) == 3:
                im = im[None]

        # Inference
        with dt[1]:
            visualize_mode = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_mode)

        # NMS
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0

            # Ensure im0 is numpy array and normalized
            if not isinstance(im0, np.ndarray):
                im0 = np.asarray(im0)
            im0 = normalize_image(im0)

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            current_bboxes = []

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Process each detection
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    bbox = [x.item() for x in xyxy]
                    current_bboxes.append(bbox)

                    if is_video:
                        # VIDEO MODE: Use tracker with multiple frames
                        track_id_preview = tracker._find_matching_plate(bbox)
                        should_do_ocr = True
                        if track_id_preview is not None:
                            should_do_ocr = tracker.should_perform_ocr(track_id_preview)

                        ocr_reading = None
                        security_status = None
                        is_in_database = False

                        if should_do_ocr:
                            ocr_reading = recognizer.recognize_plate(im0, bbox)
                            if ocr_reading and ocr_reading not in ['CROP_ERROR', 'NO_TEXT', 'OCR_ERROR']:
                                status, is_authorized, is_in_database = recognizer.check_security(ocr_reading)
                                if is_in_database:
                                    security_status = (status, is_authorized)

                        track_id, consensus_text, confidence, is_finalized, sec_status, should_continue = tracker.update(
                            bbox, ocr_reading, security_status, is_in_database
                        )

                        if sec_status[0] is not None:
                            security_status = sec_status

                        if is_finalized and consensus_text and (security_status is None or security_status[0] is None):
                            status, is_authorized, is_in_db = recognizer.check_security(consensus_text)
                            tracker.plates[track_id]['security_status'] = status
                            tracker.plates[track_id]['is_authorized'] = is_authorized
                            tracker.plates[track_id]['stop_ocr'] = True
                            security_status = (status, is_authorized)
                            LOGGER.info(f"✓ DECISION: {consensus_text} - {status}")

                        track_info = tracker.get_tracking_info(track_id)

                        # Create label for video
                        if is_finalized and consensus_text and security_status and security_status[0]:
                            status_msg, is_authorized = security_status
                            label = f'{consensus_text} ({confidence:.0%}) | {status_msg}'
                            color = COLOR_AUTHORIZED if is_authorized else COLOR_DENIED
                        elif is_finalized and consensus_text:
                            status, is_authorized, _ = recognizer.check_security(consensus_text)
                            label = f'{consensus_text} ({confidence:.0%}) | {status}'
                            color = COLOR_AUTHORIZED if is_authorized else COLOR_DENIED
                        else:
                            readings_count = track_info['readings_count'] if track_info else 0
                            label = f'Tracking... ({readings_count}/{min_readings})'
                            color = COLOR_TRACKING

                        # Save to txt for video
                        if save_txt and is_finalized and consensus_text and security_status and security_status[0]:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf, consensus_text, security_status[0]) if save_conf else (
                                cls, *xywh, consensus_text, security_status[0])
                            with open(f'{txt_path}.txt', 'a', encoding='utf-8') as f:
                                f.write(('%g ' * (len(line) - 2) + '%s %s\n').rstrip() % line)

                        # Save crop for video
                        if save_crop and is_finalized and security_status and security_status[0]:
                            crop_dir = save_dir / 'crops' / names[c]
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            crop_file = crop_dir / f'{p.stem}_{track_id}_{consensus_text}.jpg'
                            save_one_box(xyxy, imc, file=crop_file, BGR=True)

                    else:
                        # IMAGE MODE: Single OCR reading, immediate result
                        ocr_reading = recognizer.recognize_plate(im0, bbox)
                        if ocr_reading and ocr_reading not in ['CROP_ERROR', 'NO_TEXT', 'OCR_ERROR']:
                            status, is_authorized, is_in_database = recognizer.check_security(ocr_reading)
                            label = f'{ocr_reading} | {status}'
                            color = COLOR_AUTHORIZED if is_authorized else COLOR_DENIED

                            from datetime import datetime as _dt
                            _ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S')

                            # ── Always save to run folder results.txt ────────────────
                            run_result_path = save_dir / 'results.txt'
                            with open(run_result_path, 'a', encoding='utf-8') as _rf:
                                _rf.write(
                                    f"[{_ts}]\n"
                                    f"  source   : {p.name}\n"
                                    f"  plate    : {ocr_reading}\n"
                                    f"  escaped  : {ocr_reading.encode('unicode_escape').decode()}\n"
                                    f"  status   : {status}\n"
                                    f"  conf     : {conf:.2f}\n"
                                    f"  bbox     : {[int(x.item() if hasattr(x,'item') else x) for x in xyxy]}\n"
                                    f"{'-'*40}\n"
                                )

                            # ── Global plates_log.txt (appended across all runs) ─────
                            with open(Path('plates_log.txt'), 'a', encoding='utf-8') as _lf:
                                _lf.write(f"[{_ts}] plate='{ocr_reading}'  status={status}  source={p.name}\n")

                            # ── Terminal output ──────────────────────────────────────
                            sep = '=' * 52
                            LOGGER.info(sep)
                            LOGGER.info(f"  PLATE DETECTED : {ocr_reading}")
                            LOGGER.info(f"  STATUS         : {status}")
                            LOGGER.info(f"  Escaped        : {ocr_reading.encode('unicode_escape').decode()}")
                            LOGGER.info(f"  Saved to       : {run_result_path}")
                            LOGGER.info(sep)

                            # ── Optional --save-txt (YOLO label format) ───────────────
                            if save_txt:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                line = (cls, *xywh, conf, ocr_reading, status) if save_conf else (
                                    cls, *xywh, ocr_reading, status)
                                with open(f'{txt_path}.txt', 'a', encoding='utf-8') as f:
                                    f.write(('%g ' * (len(line) - 2) + '%s %s\n').rstrip() % line)

                            # ── Optional --save-crop ─────────────────────────────────
                            if save_crop:
                                crop_dir = save_dir / 'crops' / names[c]
                                crop_dir.mkdir(parents=True, exist_ok=True)
                                crop_file = crop_dir / f'{p.stem}_{ocr_reading}.jpg'
                                save_one_box(xyxy, imc, file=crop_file, BGR=True)
                        else:
                            label = f'Detecting...'
                            color = COLOR_UNKNOWN

                    # Draw on image
                    if save_img or view_img:
                        try:
                            if not hide_labels:
                                annotator.box_label(xyxy, label, color=color)
                            else:
                                annotator.box_label(xyxy, '', color=color)
                        except (AttributeError, TypeError):
                            xyxy_int = [int(x.item() if hasattr(x, 'item') else x) for x in xyxy]
                            cv2.rectangle(im0, (xyxy_int[0], xyxy_int[1]), (xyxy_int[2], xyxy_int[3]), color,
                                          line_thickness)
                            if not hide_labels and label:
                                cv2.putText(im0, label, (xyxy_int[0], max(xyxy_int[1] - 10, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Cleanup old tracks (video only)
            if is_video and tracker:
                tracker.cleanup_old_tracks(current_bboxes)

            # Display/save results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # video
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()

                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = DEFAULT_FPS, im0.shape[1], im0.shape[0]

                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * MS_CONVERSION:.1f}ms")

    # Print results
    t = tuple(x.t / seen * MS_CONVERSION for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--db-path', type=str, default=None, help='path to security database JSON file')
    parser.add_argument('--min-readings', type=int, default=MIN_READINGS_FOR_CONSENSUS,
                        help='minimum OCR readings (video only)')
    parser.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD,
                        help='consensus threshold (video only)')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
