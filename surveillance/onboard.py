import logging
import os
from datetime import datetime
from pathlib import Path

import cv2

from surveillance import config
from surveillance.db import TrainingDB
from surveillance.yolo_utils import load_model, predict_frame
from surveillance.video import sample_frames

logger = logging.getLogger(__name__)


def run_onboard(video_path: str, classname: str, username: str, init_mode: bool = False):
    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        if init_mode:
            model_path = config.DEFAULT_MODEL
            logger.info("No trained model found; downloading base model %s", model_path)
        else:
            logger.error("Model not found at %s (use --init to auto-download)", model_path)
            return 1
    if not os.path.exists(video_path):
        logger.error("Video not found: %s", video_path)
        return 1

    logger.info("Onboarding %s as '%s' from %s", username, classname, video_path)
    model = load_model(model_path)
    db = TrainingDB(config.TRAINING_DB_PATH)

    save_dir = Path(config.TRAINING_IMAGES_PATH) / username
    save_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromtimestamp(os.path.getmtime(video_path))
    saved = 0

    for frame_dt, frame in sample_frames(video_path, config.SAMPLING_FPS, start_dt):
        predictions = predict_frame(model, frame)
        matching = [p for p in predictions if p.class_name == classname]
        if not matching:
            continue

        h, w = frame.shape[:2]
        annotation_lines = []
        for det in matching:
            annotation_lines.append(
                f"{username} {det.x_center:.6f} {det.y_center:.6f} {det.width:.6f} {det.height:.6f}"
            )

        img_name = f"{username}_{frame_dt.strftime('%Y%m%d%H%M%S%f')}_{saved}.jpg"
        img_path = save_dir / img_name
        cv2.imwrite(str(img_path), frame)

        rel_path = str(Path(username) / img_name)
        dt_str = frame_dt.strftime('%Y-%m-%d %H:%M:%S')
        annotation = '\n'.join(annotation_lines)

        db.insert_image(rel_path, dt_str, classname, username, annotation, w, h)
        saved += 1

    db.close()
    logger.info("Saved %s training images of '%s' (class: %s)", saved, username, classname)
    return 0
