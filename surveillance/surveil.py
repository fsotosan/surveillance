import logging
import os
import time
import re
from collections import defaultdict

from surveillance import config
from surveillance.db import ActivityDB
from surveillance.yolo_utils import load_model, predict_frame
from surveillance.video import parse_video_filename, sample_frames

logger = logging.getLogger(__name__)

GENERIC_CLASSES = {'person', 'man', 'woman', 'dog', 'cat'}


def get_relevant_classes(detected_classes: set) -> set:
    named = detected_classes - GENERIC_CLASSES
    if named:
        return named
    return detected_classes & GENERIC_CLASSES


def run_surveillance(init_mode: bool = False):
    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        if init_mode:
            model_path = config.DEFAULT_MODEL
            logger.info("No trained model found; downloading base model %s", model_path)
        else:
            logger.error("Model not found at %s (use --init to auto-download)", model_path)
            return 1

    logger.info("Loading model from %s", model_path)
    model = load_model(model_path)
    db = ActivityDB(config.ACTIVITY_DB_PATH)
    processed = db.get_processed_files()

    logger.info("Surveillance started — watching %s", config.VIDEO_ROOT_PATH)
    logger.info("Sampling: %s fps  |  Merge window: %ss", config.SAMPLING_FPS, config.MERGE_WINDOW_SECONDS)

    while True:
        try:
            _check_and_process(model, db, processed)
        except KeyboardInterrupt:
            logger.info("Surveillance stopped.")
            break
        except Exception as e:
            logger.exception("Error in surveillance loop: %s", e)
        time.sleep(config.POLL_INTERVAL)

    db.close()
    return 0


def _get_latest_folder(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    date_folders = []
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and re.match(r'^\d{4}-\d{2}-\d{2}$', entry):
            date_folders.append(entry)
    return max(date_folders) if date_folders else None


def _wait_for_file(filepath: str, max_retries: int = 10, delay: float = 1) -> bool:
    for _ in range(max_retries):
        try:
            if not os.path.exists(filepath):
                time.sleep(delay)
                continue
            s1 = os.path.getsize(filepath)
            time.sleep(delay)
            s2 = os.path.getsize(filepath)
            if s1 == s2 and s1 > 0:
                return True
        except (OSError, FileNotFoundError):
            time.sleep(delay)
    return False


def _check_and_process(model, db: ActivityDB, processed: set):
    latest = _get_latest_folder(config.VIDEO_ROOT_PATH)
    if not latest:
        return

    folder = os.path.join(config.VIDEO_ROOT_PATH, latest)
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.mp4'):
            continue
        if filename in processed:
            continue

        filepath = os.path.join(folder, filename)
        if not _wait_for_file(filepath):
            continue

        _process_video(filepath, filename, model, db)
        db.mark_processed(filename)
        processed.add(filename)


def _process_video(filepath: str, filename: str, model, db: ActivityDB):
    info = parse_video_filename(filename)
    if not info:
        logger.warning("Skipping unparseable filename: %s", filename)
        return

    room_name, camera_number, hub_name, start_dt = info

    detections = defaultdict(list)
    for frame_dt, frame in sample_frames(filepath, config.SAMPLING_FPS, start_dt):
        for pred in predict_frame(model, frame):
            detections[pred.class_name].append(frame_dt)

    msg = f"[{room_name} cam {camera_number}] {filename}"
    if not detections:
        logger.info("%s — no detections", msg)
        return

    relevant = get_relevant_classes(set(detections.keys()))
    logger.info("%s — %s", msg, ', '.join(sorted(relevant)))

    for class_name in relevant:
        timestamps = sorted(detections[class_name])
        dt_from = timestamps[0].strftime('%Y-%m-%d %H:%M:%S')
        dt_to = timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')
        db.upsert_activity(class_name, room_name, camera_number,
                           dt_from, dt_to, config.MERGE_WINDOW_SECONDS)
