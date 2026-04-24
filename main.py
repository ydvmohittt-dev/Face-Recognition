import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pytz
import csv
from collections import defaultdict, deque
import time
import platform
import logging
import argparse
import signal
import sys
import dlib

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("attendance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Signal Handling
def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Platform Detection
def is_jetson():
    return platform.machine() == "aarch64" or os.path.exists('/etc/nv_tegra_release')

def get_jetson_model():
    if not is_jetson():
        return None
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
            if 'Orin' in content:
                return 'Orin'
            elif 'Xavier' in content:
                return 'Xavier'
            else:
                return 'Nano'
    except Exception as e:
        logger.warning(f"Failed to read Jetson model: {e}")
        return 'Unknown'

# Argument Parser
parser = argparse.ArgumentParser(description="Optimized Face Recognition Attendance System for Jetson Orin (25W)")
parser.add_argument('--data_path', type=str, default="Attendance_data", help="Path to reference images")
parser.add_argument('--csv_dir', type=str, default="Attendance_Entry", help="Directory for CSV files")
parser.add_argument('--debug_dir', type=str, default="debug_rois", help="Directory for debug ROIs")
parser.add_argument('--match_tolerance', type=float, default=0.45, help="Face match tolerance")
parser.add_argument('--confidence_threshold', type=float, default=0.45, help="DNN confidence threshold")
parser.add_argument('--max_encodings_per_person', type=int, default=5, help="Max reference images per person")
parser.add_argument('--headless', action='store_true', help="Run without UI")
parser.add_argument('--test', action='store_true', help="Test mode without camera")
parser.add_argument('--use_hog', action='store_true', help="Use HOG model instead of CNN")
args = parser.parse_args()

# Dynamic Configuration
JETSON_PLATFORM = is_jetson()
JETSON_MODEL = get_jetson_model() if JETSON_PLATFORM else None

if JETSON_PLATFORM:
    if JETSON_MODEL == 'Orin':
        FRAME_WIDTH = 640  # Wider window
        FRAME_HEIGHT = 360
        DETECTION_INTERVAL = 3  # Balanced for 25W
        STABILITY_FRAMES = 4
        FACE_MODEL = "hog" if args.use_hog or not dlib.DLIB_USE_CUDA else "cnn"
        RESIZE_SCALE = 0.5
        NO_FACE_ADAPTIVE_MULTIPLIER = 3  # Check 3x less frequently when no faces
        DISPLAY_SCALE = 1.5  # Scale window to ~960x540
    else:
        FRAME_WIDTH = 320
        FRAME_HEIGHT = 240
        DETECTION_INTERVAL = 5
        STABILITY_FRAMES = 4
        FACE_MODEL = "hog"
        RESIZE_SCALE = 0.75
        NO_FACE_ADAPTIVE_MULTIPLIER = 3
        DISPLAY_SCALE = 2.0
else:
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    DETECTION_INTERVAL = 3
    STABILITY_FRAMES = 4
    FACE_MODEL = "hog"
    RESIZE_SCALE = 0.5
    NO_FACE_ADAPTIVE_MULTIPLIER = 3
    DISPLAY_SCALE = 2.0

DATA_PATH = os.path.abspath(args.data_path)
CSV_DIR = os.path.abspath(args.csv_dir)
DEBUG_DIR = os.path.abspath(args.debug_dir)
MATCH_TOLERANCE = args.match_tolerance
CONFIDENCE_THRESHOLD = args.confidence_threshold
MAX_ENCODINGS_PER_PERSON = args.max_encodings_per_person
HEADLESS = args.headless

# Global Variables
attendance_set = set()
face_tracker = defaultdict(lambda: {
    'name': 'UNKNOWN', 'count': 0, 'last_pos': None,
    'distances': deque(maxlen=STABILITY_FRAMES),
    'last_seen': time.time(), 'last_rect': None,
    'rect_history': deque(maxlen=5)
})

frame_times = deque(maxlen=60)
CUDA_STATUS = {"dlib": False, "opencv": False}

# Helper Functions
def cleanup_stale_trackers(timeout=7.0):
    current_time = time.time()
    stale_ids = [fid for fid, tracker in face_tracker.items()
                 if current_time - tracker['last_seen'] > timeout]
    for fid in stale_ids:
        del face_tracker[fid]

def get_date_string():
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y_%m_%d")

def get_timestamp_for_display():
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")

def load_reference_images(data_path=DATA_PATH):
    images, class_names = [], []
    if not os.path.exists(data_path):
        logger.error(f"Data path '{data_path}' does not exist.")
        return images, class_names

    person_dirs = [d for d in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]
    logger.info(f"Found {len(person_dirs)} persons: {person_dirs}")

    total_images = 0
    for person in person_dirs:
        person_dir = os.path.join(data_path, person)
        person_files = [f for f in os.listdir(person_dir)
                        if not f.startswith(".") and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        person_files = person_files[:MAX_ENCODINGS_PER_PERSON]
        person_images = 0

        for fname in person_files:
            fullpath = os.path.join(person_dir, fname)
            img = cv2.imread(fullpath)
            if img is None:
                logger.warning(f"Could not read {fullpath}")
                continue
            images.append(img)
            class_names.append(person)
            person_images += 1
            total_images += 1
        logger.info(f"  • {person}: {person_images} images")

    logger.info(f"Loaded {total_images} total images for {len(person_dirs)} persons")
    return images, class_names

def identify_encodings(images, class_names):
    logger.info("Generating face encodings...")
    encode_dict = defaultdict(list)
    total = len(images)

    for idx, (img, name) in enumerate(zip(images, class_names), 1):
        if idx % 5 == 0 or idx == total:
            logger.debug(f"Processing: {idx}/{total} images...")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale)

        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            encode_dict[name].append(encodings[0])
        else:
            logger.warning(f"No face found for {name} in image {idx}")

    logger.info(f"Successfully generated encodings for {len(encode_dict)} persons")
    return dict(encode_dict)

def match_face(face_encoding, encode_dict, tolerance=MATCH_TOLERANCE):
    min_avg_distance = float('inf')
    best_name = "UNKNOWN"

    for known_name, known_encodings in encode_dict.items():
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        avg_distance = np.mean(distances)
        if avg_distance < min_avg_distance and avg_distance <= tolerance:
            min_avg_distance = avg_distance
            best_name = known_name

    return best_name, min_avg_distance

def load_daily_attendance(csv_filepath):
    daily_attendance = set()
    if os.path.exists(csv_filepath):
        try:
            with open(csv_filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                today = get_date_string().replace("_", "-")
                for row in reader:
                    if row and len(row) >= 3 and row[2] == today:
                        daily_attendance.add(row[0])
        except Exception as e:
            logger.error(f"Could not read {csv_filepath}: {e}")
    return daily_attendance

def mark_attendance(name, csv_filepath):
    if name == "UNKNOWN" or name in attendance_set:
        return False
    attendance_set.add(name)
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    date_str = get_date_string().replace("_", "-")
    time_str = now.strftime('%H:%M:%S')
    try:
        with open(csv_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, time_str, date_str])
        logger.info(f"Marked {name} at {time_str}")
        return True
    except Exception as e:
        logger.error(f"Could not write to CSV: {e}")
        return False

def detect_faces_dnn(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_rects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_rects.append((x1, y1, x2, y2))
    return face_rects

def smooth_rect(rect, history):
    if not history:
        return rect
    rects = list(history) + [rect]
    x1 = int(np.mean([r[0] for r in rects]))
    y1 = int(np.mean([r[1] for r in rects]))
    x2 = int(np.mean([r[2] for r in rects]))
    y2 = int(np.mean([r[3] for r in rects]))
    return (x1, y1, x2, y2)

def draw_info_panel(frame, encode_dict, fps, detected_faces):
    h, w = frame.shape[:2]
    panel_height = 50
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    cv2.putText(panel, f"Time: {get_timestamp_for_display()}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(panel, f"Reg: {len(encode_dict)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    
    # CUDA Status display
    cuda_color = (0, 255, 0) if (CUDA_STATUS["dlib"] or CUDA_STATUS["opencv"]) else (0, 165, 255)
    cuda_text = "CUDA: "
    if CUDA_STATUS["dlib"] and CUDA_STATUS["opencv"]:
        cuda_text += "ON (dlib+OpenCV)"
    elif CUDA_STATUS["dlib"]:
        cuda_text += "ON (dlib)"
    elif CUDA_STATUS["opencv"]:
        cuda_text += "ON (OpenCV)"
    else:
        cuda_text += "OFF"
    cv2.putText(panel, cuda_text, (int(w * 0.35), 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cuda_color, 1)
    
    cv2.putText(panel, f"FPS: {fps:.1f}", (w - 80, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    cv2.putText(panel, f"Faces: {detected_faces}", (w - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    return panel

def draw_attendance_list(frame, attendance_set, max_display=3):
    if not attendance_set:
        return
    h, w = frame.shape[:2]
    list_width = 200  # Wider for larger window
    list_height = min(30 + len(list(attendance_set)[:max_display]) * 15, 90)
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - list_width - 5, 60), (w - 5, 60 + list_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Recent:", (w - list_width, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    for idx, name in enumerate(list(attendance_set)[:max_display], 1):
        y_pos = 75 + idx * 15
        cv2.putText(frame, f"{idx}. {name}", (w - list_width + 5, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)

def main():
    logger.info("=" * 60)
    logger.info(f"{' ' * 10}JETSON {JETSON_MODEL or 'UNKNOWN'} ATTENDANCE SYSTEM")
    logger.info("=" * 60)

    # Check dlib CUDA
    if dlib.DLIB_USE_CUDA:
        logger.info(f"✓ dlib CUDA enabled with {dlib.cuda.get_num_devices()} devices")
        CUDA_STATUS["dlib"] = True
    else:
        logger.warning("dlib CUDA not enabled. Using HOG model. Reinstall dlib with CUDA for better performance.")
        CUDA_STATUS["dlib"] = False

    # Platform info
    if JETSON_PLATFORM:
        logger.info(f"Platform: NVIDIA Jetson {JETSON_MODEL}")
        logger.info(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT} (Display scaled: {int(FRAME_WIDTH*DISPLAY_SCALE)}x{int((FRAME_HEIGHT+50)*DISPLAY_SCALE)})")
        logger.info(f"Detection interval: Every {DETECTION_INTERVAL} frames")
        logger.info(f"Face recognition model: {FACE_MODEL.upper()}")
        logger.info("Running in 25W mode for optimal performance")

    # Check CPU governor
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            governor = f.read().strip()
        logger.info(f"CPU governor: {governor}")
        if governor != 'performance':
            logger.warning("Low-power mode detected. Run 'sudo nvpmodel -m 0' for best performance.")
    except:
        logger.warning("Could not check CPU governor.")

    # Create folders
    try:
        os.makedirs(CSV_DIR, exist_ok=True)
        os.makedirs(DEBUG_DIR, exist_ok=True)
        logger.info("Directories verified")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return

    # Prepare CSV
    today = get_date_string()
    csv_filepath = os.path.join(CSV_DIR, f"Attendance_{today}.csv")
    if not os.path.exists(csv_filepath):
        try:
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time", "Date"])
            logger.info(f"Created new attendance file: {csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to create CSV: {e}")
            return
    else:
        logger.info(f"Using existing attendance file: {csv_filepath}")

    # Load existing attendance
    global attendance_set
    attendance_set = load_daily_attendance(csv_filepath)
    if attendance_set:
        logger.info(f"Loaded {len(attendance_set)} existing attendance records")
        logger.info(f"Already marked: {', '.join(sorted(attendance_set))}")

    # Load encodings
    logger.info("\n" + "=" * 60)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 60)
    images, class_names = load_reference_images()
    if not images:
        logger.error("No training images found. Run initial_data_capture.py first.")
        return

    encode_dict = identify_encodings(images, class_names)
    if not encode_dict:
        logger.error("No valid encodings generated. Exiting.")
        return

    logger.info("=" * 60)

    # Load DNN model
    logger.info("Loading face detection model...")
    model_path = os.path.join(os.path.dirname(__file__), 'opencv_face_detector_uint8.pb')
    config_path = os.path.join(os.path.dirname(__file__), 'opencv_face_detector.pbtxt')
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        logger.error("DNN model files not found:")
        logger.error(f"  Expected: {model_path}")
        logger.error(f"  Expected: {config_path}")
        return

    try:
        net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("✓ CUDA backend enabled")
            CUDA_STATUS["opencv"] = True
            try:
                cv2.cuda.setBufferPoolUsage(True)
                cv2.cuda.setBufferPoolConfig(cv2.cuda.getDevice(), 1024 * 1024 * 32, 2)
                logger.info("✓ GPU memory pool configured")
            except Exception as e:
                logger.warning(f"Could not configure GPU memory pool: {e}")
        else:
            logger.info("Using CPU backend for DNN")
            CUDA_STATUS["opencv"] = False
    except Exception as e:
        logger.error(f"Failed to load DNN model: {e}")
        return

    # Camera setup
    logger.info("Initializing camera...")
    cap = None
    if args.test:
        logger.info("Test mode: Skipping camera initialization")
    else:
        if JETSON_PLATFORM:
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id=0 ! "
                f"video/x-raw(memory:NVMM), width={FRAME_WIDTH}, height={FRAME_HEIGHT}, "
                f"format=NV12, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                logger.warning("GStreamer pipeline failed, trying standard camera...")
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)

        if cap:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap or not cap.isOpened():
            logger.error("Cannot open camera.")
            return

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera resolution: {actual_width}x{actual_height}")

    # Initialize resizable window
    if not HEADLESS and not args.test:
        cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Attendance System", int(FRAME_WIDTH * DISPLAY_SCALE), int((FRAME_HEIGHT + 50) * DISPLAY_SCALE))

    logger.info("\n" + "=" * 60)
    logger.info("SYSTEM READY - Starting attendance monitoring")
    logger.info("=" * 60)
    if not args.test:
        logger.info("Position yourself in front of the camera")
    logger.info("Press 'q' to quit (or Ctrl+C)")
    logger.info("=" * 60 + "\n")

    frame_count = 0
    cleanup_counter = 0
    no_face_counter = 0
    last_face_count = 0

    while True:
        try:
            frame_start = time.time()

            if args.test:
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break

            frame_disp = frame.copy()
            rgb_frame = None
            face_rects = []

            # FIXED: Adaptive detection interval - ALWAYS runs, never stops
            # When no faces detected, checks less frequently to save resources
            if no_face_counter >= 10:
                # No faces for a while - check less frequently (3x interval)
                detection_interval = DETECTION_INTERVAL * NO_FACE_ADAPTIVE_MULTIPLIER
            else:
                # Faces present or recently seen - normal interval
                detection_interval = DETECTION_INTERVAL

            # Always run detection at the appropriate interval
            if frame_count % detection_interval == 0:
                frame_small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
                face_rects_small = detect_faces_dnn(frame_small, net)
                scale_back = 1.0 / RESIZE_SCALE
                face_rects = [(int(x * scale_back), int(y * scale_back),
                               int(X * scale_back), int(Y * scale_back))
                              for (x, y, X, Y) in face_rects_small]
                last_face_count = len(face_rects)
                
                # Update counter based on detection results
                if last_face_count > 0:
                    no_face_counter = 0  # Reset when faces found
                    logger.debug(f"Detected {last_face_count} face(s)")
                else:
                    no_face_counter = min(no_face_counter + 1, 100)  # Cap at 100
                    if no_face_counter % 30 == 0:
                        logger.debug(f"No faces detected for {no_face_counter} cycles (checking every {detection_interval} frames)")

            cleanup_counter += 1
            if cleanup_counter % 50 == 0:
                cleanup_stale_trackers()

            face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in face_rects]
            if face_locations and rgb_frame is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detect_time = time.time() - frame_start
            encode_start = time.time()

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model=FACE_MODEL) if face_locations else []
            encode_time = time.time() - encode_start

            for idx, (left, top, right, bottom) in enumerate(face_rects):
                if idx >= len(face_encodings):
                    continue

                face_encoding = face_encodings[idx]
                display_name = "UNKNOWN"
                color = (0, 0, 255)
                confidence_text = ""

                if face_encoding.size > 0:
                    candidate_name, distance = match_face(face_encoding, encode_dict)
                    face_center = ((left + right) // 2, (top + bottom) // 2)
                    face_id = None
                    min_dist = 100

                    for fid, tracker in face_tracker.items():
                        if tracker['last_pos']:
                            dist = np.linalg.norm(np.array(face_center) - np.array(tracker['last_pos']))
                            if dist < min_dist:
                                min_dist = dist
                                face_id = fid

                    if face_id is None:
                        face_id = f"{frame_count}_{face_center[0]}_{face_center[1]}"

                    tracker = face_tracker[face_id]
                    tracker['last_pos'] = face_center
                    tracker['last_seen'] = time.time()
                    tracker['distances'].append(distance)
                    tracker['last_rect'] = (left, top, right, bottom)
                    tracker['rect_history'].append((left, top, right, bottom))

                    if candidate_name != "UNKNOWN" and (tracker['name'] == "UNKNOWN" or tracker['name'] == candidate_name):
                        tracker['count'] = tracker['count'] + 1 if tracker['name'] == candidate_name else 1
                        tracker['name'] = candidate_name
                    else:
                        tracker['count'] = 0
                        tracker['name'] = "UNKNOWN"

                    display_name = tracker['name'] if (tracker['count'] >= STABILITY_FRAMES and
                                                       np.mean(tracker['distances']) <= MATCH_TOLERANCE) else "UNKNOWN"
                    confidence = max(0, (1 - np.mean(tracker['distances'])) * 100)
                    confidence_text = f" ({confidence:.0f}%)"
                    color = (0, 255, 0) if display_name != "UNKNOWN" else (0, 0, 255)

                    if display_name != "UNKNOWN":
                        marked = mark_attendance(display_name, csv_filepath)
                        if marked and not HEADLESS:
                            cv2.rectangle(frame_disp, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 0), 2)

                    if display_name == "UNKNOWN" and frame_count % 10 == 0:
                        roi = frame[max(0, top-20):bottom+20, max(0, left-20):right+20]
                        if roi.size > 0:
                            debug_path = os.path.join(DEBUG_DIR, f"unknown_{frame_count}_{left}_{top}.jpg")
                            cv2.imwrite(debug_path, roi, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            logger.debug(f"Saved unknown face ROI to {debug_path}")

                smoothed_rect = smooth_rect((left, top, right, bottom), tracker['rect_history'])
                left, top, right, bottom = smoothed_rect
                cv2.rectangle(frame_disp, (left, top), (right, bottom), color, 1)
                label_text = display_name + confidence_text if confidence_text else display_name
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame_disp, (left, bottom - text_height - 6), (left + text_width + 6, bottom), color, cv2.FILLED)
                cv2.putText(frame_disp, label_text, (left + 3, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for fid, tracker in list(face_tracker.items()):
                if tracker['last_seen'] > time.time() - 2.0 and tracker['last_rect']:
                    left, top, right, bottom = smooth_rect(tracker['last_rect'], tracker['rect_history'])
                    color = (0, 255, 0) if tracker['name'] != "UNKNOWN" else (0, 0, 255)
                    cv2.rectangle(frame_disp, (left, top), (right, bottom), color, 1)
                    label_text = tracker['name']
                    if tracker['distances']:
                        confidence = max(0, (1 - np.mean(tracker['distances'])) * 100)
                        label_text += f" ({confidence:.0f}%)"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame_disp, (left, bottom - text_height - 6), (left + text_width + 6, bottom), color, cv2.FILLED)
                    cv2.putText(frame_disp, label_text, (left + 3, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            fps = 1.0 / np.mean(frame_times) if frame_times else 0

            if not HEADLESS and not args.test and frame_count % 2 == 0:
                info_panel = draw_info_panel(frame_disp, encode_dict, fps, last_face_count)
                frame_with_panel = np.vstack([info_panel, frame_disp])
                draw_attendance_list(frame_with_panel, attendance_set)
                # Scale display for wider window
                display_frame = cv2.resize(frame_with_panel, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Attendance System", display_frame)

            frame_count += 1
            logger.debug(f"Frame {frame_count}: Detect {detect_time*1000:.1f}ms, Encode {encode_time*1000:.1f}ms, FPS {fps:.1f}")

            if not HEADLESS and cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit command received")
                break

            if args.test and frame_count >= 10:
                break

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(0.1)

    if cap:
        cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    logger.info("\n" + "=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Platform: {'Jetson ' + JETSON_MODEL if JETSON_PLATFORM else 'PC'}")
    logger.info(f"Total frames processed: {frame_count}")
    if frame_times:
        logger.info(f"Average FPS: {1.0 / np.mean(frame_times):.1f}")
    logger.info(f"Attendance marked today: {len(attendance_set)}")
    if attendance_set:
        logger.info(f"Present: {', '.join(sorted(attendance_set))}")
    logger.info(f"CSV file: {csv_filepath}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
