import logging
import os
import time
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2

from surveillance import config
from surveillance.db import ActivityDB
from surveillance.yolo_utils import load_model, predict_frame
from surveillance.video import parse_video_filename, sample_frames

logger = logging.getLogger(__name__)

GENERIC_CLASSES = {'person', 'man', 'woman', 'dog', 'cat'}

_LABEL_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def get_relevant_classes(detected_classes: set) -> set:
    named = detected_classes - GENERIC_CLASSES
    if named:
        return named
    return detected_classes & GENERIC_CLASSES


def run_surveillance(init_mode: bool = False, bulk_mode: bool = False):
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

    if bulk_mode:
        logger.info("Bulk mode — processing all video folders...")
        try:
            _check_and_process_all(model, db, processed)
        except Exception as e:
            logger.exception("Error during bulk processing: %s", e)
        db.close()
        return 0

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


def _get_latest_folder(root: str) -> Optional[str]:
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


def _check_and_process_all(model, db: ActivityDB, processed: set):
    root = config.VIDEO_ROOT_PATH
    if not os.path.isdir(root):
        logger.warning("Video root not found: %s", root)
        return

    date_folders = sorted([
        entry for entry in os.listdir(root)
        if os.path.isdir(os.path.join(root, entry))
        and re.match(r'^\d{4}-\d{2}-\d{2}$', entry)
    ])

    for date_folder in date_folders:
        folder = os.path.join(root, date_folder)
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith('.mp4'):
                continue
            if filename in processed:
                continue

            filepath = os.path.join(folder, filename)
            if not _wait_for_file(filepath):
                logger.warning("Skipping unstable file: %s", filename)
                continue

            _process_video(filepath, filename, model, db)
            db.mark_processed(filename)
            processed.add(filename)

    logger.info("Bulk processing complete.")


def _save_labeled_frame(frame, frame_dt, room_name: str,
                        relevant_classes: list,
                        all_classes: list,
                        predictions: list) -> str:
    class_to_color = {
        name: _LABEL_COLORS[i % len(_LABEL_COLORS)]
        for i, name in enumerate(all_classes)
    }
    h, w = frame.shape[:2]

    for det in predictions:
        color = class_to_color[det.class_name]
        x1 = int((det.x_center - det.width / 2) * w)
        y1 = int((det.y_center - det.height / 2) * h)
        x2 = int((det.x_center + det.width / 2) * w)
        y2 = int((det.y_center + det.height / 2) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = det.class_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    date_folder = frame_dt.strftime('%Y-%m-%d')
    timestamp_str = frame_dt.strftime('%Y%m%d%H%M%S')
    classes_str = '-'.join(relevant_classes)
    filename = f"{room_name}_{timestamp_str}_{classes_str}.png"

    save_dir = Path(config.ACTIVITY_ROOT_PATH) / date_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / filename), frame)

    return f"{date_folder}/{filename}"


def _process_video(filepath: str, filename: str, model, db: ActivityDB):
    info = parse_video_filename(filename)
    if not info:
        logger.warning("Skipping unparseable filename: %s", filename)
        return

    room_name, camera_number, hub_name, start_dt = info

    first_for_class = {}
    last_for_class = {}
    all_class_names = set()

    for frame_dt, frame in sample_frames(filepath, config.SAMPLING_FPS, start_dt):
        preds = predict_frame(model, frame)
        if not preds:
            continue
        for p in preds:
            all_class_names.add(p.class_name)
            if p.class_name not in first_for_class:
                first_for_class[p.class_name] = (frame_dt, frame.copy(), preds)
            last_for_class[p.class_name] = (frame_dt, frame.copy(), preds)

    msg = f"[{room_name} cam {camera_number}] {filename}"
    if not all_class_names:
        logger.info("%s — no detections", msg)
        return

    relevant = get_relevant_classes(all_class_names)
    logger.info("%s — %s", msg, ', '.join(sorted(relevant)))

    all_sorted = sorted(all_class_names)
    relevant_sorted = sorted(relevant)

    for class_name in relevant:
        first_frame_dt, first_frame, first_preds = first_for_class[class_name]
        last_frame_dt, last_frame, last_preds = last_for_class[class_name]

        dt_from = first_frame_dt.strftime('%Y-%m-%d %H:%M:%S')
        dt_to = last_frame_dt.strftime('%Y-%m-%d %H:%M:%S')

        img_first = _save_labeled_frame(
            first_frame, first_frame_dt, room_name,
            relevant_sorted, all_sorted, first_preds,
        )
        img_last = _save_labeled_frame(
            last_frame, last_frame_dt, room_name,
            relevant_sorted, all_sorted, last_preds,
        )

        db.upsert_activity(class_name, room_name, camera_number,
                           dt_from, dt_to, img_first, img_last,
                           config.MERGE_WINDOW_SECONDS)
