import cv2
import os
import face_recognition
import numpy as np
import platform
from datetime import datetime
import pytz
import logging
import argparse
import signal
import sys
import shutil
import dlib

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data_capture.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Signal Handling for Clean Exit
def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Argument Parser
parser = argparse.ArgumentParser(description="Optimized Data Capture for Attendance System on Jetson Orin")
parser.add_argument('--base_path', type=str, default="Attendance_data", help="Path to store reference images")
parser.add_argument('--camera_id', type=int, default=0, help="Camera index or GStreamer pipeline")
parser.add_argument('--max_images', type=int, default=10, help="Max images per person")
parser.add_argument('--frame_width', type=int, default=480, help="Camera frame width")
parser.add_argument('--frame_height', type=int, default=360, help="Camera frame height")
parser.add_argument('--headless', action='store_true', help="Run without UI")
parser.add_argument('--test', action='store_true', help="Test mode without camera")
args = parser.parse_args()

# Configuration
BASE_PATH = os.path.abspath(args.base_path)
CAMERA_ID = args.camera_id
MAX_IMAGES = args.max_images
FRAME_WIDTH = args.frame_width
FRAME_HEIGHT = args.frame_height
HEADLESS = args.headless
DETECTION_INTERVAL = 2
CONFIDENCE_THRESHOLD = 0.45
RESIZE_SCALE = 0.4

# Platform Detection
def is_jetson():
    return platform.system() == "Linux" and "tegra" in platform.uname().release.lower()

IS_JETSON = is_jetson()
FACE_MODEL = "cnn" if IS_JETSON else "hog"

# Check Camera Availability
def check_camera_availability():
    logger.info("Checking camera availability...")
    for cam_id in range(3):
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                logger.info(f"Camera found at index {cam_id}")
                return cam_id
    logger.error("No working camera found")
    return None

# DNN Face Detection (Aligned with main.py)
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

# List Existing Persons
def list_existing_persons(base_path=BASE_PATH):
    if not os.path.exists(base_path):
        return []
    persons = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]
    if persons:
        logger.info(f"Existing persons in database ({len(persons)}):")
        for i, person in enumerate(sorted(persons), 1):
            person_dir = os.path.join(base_path, person)
            img_count = len([f for f in os.listdir(person_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            logger.info(f"  {i}. {person} ({img_count} images)")
    return persons

# Initial Data Capture
def initial_data_capture(camera_id=CAMERA_ID, max_images=MAX_IMAGES, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
    os.makedirs(BASE_PATH, exist_ok=True)
    existing_persons = list_existing_persons()

    # Get person name
    name = input("Please Enter your name: ").strip().upper()
    if not name:
        logger.error("Name cannot be empty.")
        return

    person_dir = os.path.join(BASE_PATH, name)
    if name in existing_persons:
        logger.warning(f"'{name}' already exists in the database.")
        action = input("Choose action - [A]dd more images, [R]eplace all, [C]ancel: ").strip().upper()
        if action == 'C':
            logger.info("Operation canceled.")
            return
        elif action == 'R':
            logger.info(f"Removing existing images for '{name}'...")
            try:
                shutil.rmtree(person_dir)
                os.makedirs(person_dir, exist_ok=True)
                logger.info("Existing images removed.")
            except Exception as e:
                logger.error(f"Error removing existing images: {e}")
                return
        elif action == 'A':
            logger.info(f"Adding more images to existing collection for '{name}'.")
        else:
            logger.error("Invalid choice. Operation canceled.")
            return
    else:
        os.makedirs(person_dir, exist_ok=True)

    # Load DNN model
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
            logger.info("✓ CUDA backend enabled for DNN")
        else:
            logger.info("Using CPU backend for DNN")
    except Exception as e:
        logger.error(f"Failed to load DNN model: {e}")
        return

    # Camera setup
    if args.test:
        logger.info("Test mode: Skipping camera initialization")
        camera = None
    else:
        if IS_JETSON:
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={camera_id} ! "
                f"video/x-raw(memory:NVMM), width={frame_width}, height={frame_height}, "
                f"format=NV12, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not camera.isOpened():
                logger.warning("GStreamer pipeline failed, trying standard camera...")
                camera = cv2.VideoCapture(camera_id)
        else:
            camera = cv2.VideoCapture(camera_id)

        if not camera.isOpened():
            detected_cam = check_camera_availability()
            if detected_cam is not None:
                use_detected = input(f"Use detected camera at index {detected_cam}? (y/n): ").strip().lower()
                if use_detected == 'y':
                    camera = cv2.VideoCapture(detected_cam)
                else:
                    logger.error("Cannot open camera.")
                    return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera resolution: {actual_width}x{actual_height} (requested: {frame_width}x{frame_height})")

    logger.info(f"\n{'='*60}")
    logger.info(f"Platform: {'Jetson' if IS_JETSON else 'Windows'}")
    logger.info(f"Detection model: {FACE_MODEL.upper()}")
    logger.info(f"Max images: {max_images}")
    logger.info(f"{'='*60}")
    logger.info("INSTRUCTIONS:")
    logger.info("  • Press SPACE to capture an image (single face required)")
    logger.info("  • Press 'q' to quit early")
    logger.info("  • Press 'r' to retake the last captured image")
    logger.info("  • Ensure good lighting and face the camera directly")
    logger.info("  • Vary your pose slightly between captures")
    logger.info(f"{'='*60}\n")

    img_count = 0
    last_saved_filename = None
    capture_times = []
    start_time = datetime.now()
    frame_count = 0
    last_face_count = 0

    while img_count < max_images:
        try:
            if args.test:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                ret = True
            else:
                ret, frame = camera.read()
                if not ret:
                    logger.warning("Failed to grab frame.")
                    break

            frame_count += 1
            display_frame = frame.copy()
            faces = []

            # Skip detection if no faces recently or not on interval
            if frame_count % DETECTION_INTERVAL == 0 or last_face_count > 0:
                frame_small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
                faces_small = detect_faces_dnn(frame_small, net)
                scale_back = 1.0 / RESIZE_SCALE
                faces = [(int(x * scale_back), int(y * scale_back), int(X * scale_back), int(Y * scale_back))
                         for (x, y, X, Y) in faces_small]
                last_face_count = len(faces)

            # Draw status bar
            if not HEADLESS:
                status_bar = np.zeros((50, frame.shape[1], 3), dtype=np.uint8)
                cv2.putText(status_bar, f"Progress: {img_count}/{max_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(status_bar, f"Person: {name}", (frame.shape[1] - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Process face detection results
            face_status = ""
            face_color = (0, 0, 255)
            if faces:
                if len(faces) == 1:
                    x1, y1, x2, y2 = faces[0]
                    roi_h, roi_w = y2 - y1, x2 - x1
                    if roi_h < 100 or roi_w < 100:
                        face_status = "Face too small - move closer"
                        face_color = (0, 165, 255)
                        if not HEADLESS:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), face_color, 2)
                    else:
                        face_status = "Ready to capture! Press SPACE"
                        face_color = (0, 255, 0)
                        if not HEADLESS:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), face_color, 3)
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.circle(display_frame, (center_x, center_y), 5, face_color, -1)
                else:
                    face_status = f"{len(faces)} faces detected - need exactly 1"
                    if not HEADLESS:
                        for (x1, y1, x2, y2) in faces:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), face_color, 2)
            else:
                face_status = "No face detected"
                last_face_count = 0

            if not HEADLESS:
                cv2.putText(display_frame, face_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                combined_frame = np.vstack([status_bar, display_frame])
                cv2.imshow("Camera Preview", combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if len(faces) == 1 and roi_h >= 100 and roi_w >= 100:
                    img_count += 1
                    filename = os.path.join(person_dir, f"{name}_{img_count}.jpg")
                    cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    last_saved_filename = filename
                    capture_times.append(datetime.now())
                    logger.info(f"[{img_count}/{max_images}] Saved {os.path.basename(filename)}")
                    if img_count >= max_images:
                        logger.info(f"Reached max image count ({max_images})!")
                        break
                else:
                    logger.warning(f"Cannot capture: Detected {len(faces)} faces or face too small.")

            elif key == ord('r') and last_saved_filename and img_count > 0:
                try:
                    if os.path.exists(last_saved_filename):
                        os.remove(last_saved_filename)
                        img_count -= 1
                        if capture_times:
                            capture_times.pop()
                        logger.info(f"Deleted last image. Count reset to {img_count}/{max_images}")
                        last_saved_filename = None
                except Exception as e:
                    logger.error(f"Error deleting last image: {e}")

            elif key == ord('q'):
                logger.info("Quitting early...")
                break

        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
            time.sleep(0.1)

    if camera:
        camera.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\n{'='*60}")
    logger.info("CAPTURE SESSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  • Person: {name}")
    logger.info(f"  • Images captured: {img_count}/{max_images}")
    logger.info(f"  • Session duration: {duration:.1f} seconds")
    if img_count > 1 and capture_times:
        avg_interval = (capture_times[-1] - capture_times[0]).total_seconds() / (len(capture_times) - 1)
        logger.info(f"  • Average capture interval: {avg_interval:.1f} seconds")
    logger.info(f"  • Saved to: {person_dir}")
    logger.info(f"{'='*60}")
    if img_count < max_images:
        logger.info(f"Note: Only {img_count} images captured. Consider capturing more for better recognition.")
    else:
        logger.info("Data capture complete! You can now run the attendance system.")

# Batch Capture
def batch_capture(max_images=MAX_IMAGES):
    logger.info("=== BATCH CAPTURE MODE ===")
    logger.info("Capture images for multiple persons without restarting.")
    while True:
        initial_data_capture(max_images=max_images)
        another = input("\nCapture images for another person? (y/n): ").strip().lower()
        if another != 'y':
            logger.info("Batch capture session ended.")
            break

if __name__ == "__main__":
    # Check dlib CUDA
    if IS_JETSON and not dlib.DLIB_USE_CUDA:
        logger.warning("dlib CUDA not enabled. Performance may be reduced. Reinstall dlib with CUDA.")
    logger.info("=== ATTENDANCE SYSTEM - DATA CAPTURE ===")
    logger.info("1. Single person capture")
    logger.info("2. Batch capture (multiple persons)")
    logger.info("3. Exit")
    choice = input("\nSelect option (1-3): ").strip()
    if choice == '1':
        initial_data_capture()
    elif choice == '2':
        batch_capture()
    elif choice == '3':
        logger.info("Exiting...")
    else:
        logger.error("Invalid choice. Exiting...")