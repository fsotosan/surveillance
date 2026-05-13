import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from surveillance import config
from surveillance.db import TrainingDB

logger = logging.getLogger(__name__)


def run_train(epochs: int = 100, imgsz: int = 640, batch: int = 16, export_only: bool = False):
    db = TrainingDB(config.TRAINING_DB_PATH)
    images = db.get_all_images()
    usernames = db.get_usernames()
    db.close()

    if not images:
        logger.error("No training data found in training database.")
        return 1
    if len(usernames) < 1:
        logger.error("No unique usernames found.")
        return 1

    logger.info("Training set: %s images, %s classes: %s", len(images), len(usernames), usernames)

    dataset_dir = Path(config.TRAINING_IMAGES_PATH).parent / 'dataset'
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ('train', 'val'):
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_set = images[:split_idx]
    val_set = images[split_idx:]

    class_to_id = {name: idx for idx, name in enumerate(usernames)}
    images_root = Path(config.TRAINING_IMAGES_PATH)

    def _export(split_name, data):
        for img in data:
            src = images_root / img['relative_path']
            if not src.exists():
                continue
            out_name = f"{img['id']:06d}.jpg"
            shutil.copy2(str(src), str(dataset_dir / 'images' / split_name / out_name))

            lines = []
            for line in img['annotation'].strip().split('\n'):
                parts = line.split()
                if len(parts) == 5:
                    uname = parts[0]
                    if uname in class_to_id:
                        cid = class_to_id[uname]
                        lines.append(f"{cid} {' '.join(parts[1:])}")
            if lines:
                label_path = dataset_dir / 'labels' / split_name / f"{img['id']:06d}.txt"
                label_path.write_text('\n'.join(lines))

    _export('train', train_set)
    _export('val', val_set)

    data_yaml = {
        'train': str(dataset_dir / 'images' / 'train'),
        'val': str(dataset_dir / 'images' / 'val'),
        'nc': len(usernames),
        'names': usernames,
    }
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    logger.info("Dataset exported to %s", dataset_dir)

    if export_only:
        _generate_labeled_preview(dataset_dir, usernames)
        logger.info("--export-only set; skipping training. Review dataset at %s", dataset_dir)
        return 0

    model_path = Path(config.MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        backup_name = f"best_{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        backup_path = model_path.parent / backup_name
        shutil.copy2(str(model_path), str(backup_path))
        logger.info("Backed up old model → %s", backup_path)

    from ultralytics import YOLO
    from surveillance.yolo_utils import _resolve_device

    base = config.DEFAULT_MODEL
    device = _resolve_device()
    logger.info("Starting training from base model: %s on %s", base, device)
    yolo = YOLO(base)
    yolo.to(device)
    yolo.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(model_path.parent),
        name='_train',
        exist_ok=True,
        verbose=True,
    )

    trained_best = model_path.parent / '_train' / 'weights' / 'best.pt'
    if trained_best.exists():
        shutil.copy2(str(trained_best), str(model_path))
        shutil.rmtree(str(model_path.parent / '_train'), ignore_errors=True)
        logger.info("Trained model saved to %s", model_path)
    else:
        logger.error("Training failed: best.pt not found in output.")
        return 1

    return 0


def _generate_labeled_preview(dataset_dir: Path, class_names: list):
    import cv2

    preview_dir = dataset_dir / 'labeled_preview'
    preview_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]

    count = 0
    for split in ('train', 'val'):
        img_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            label_path = label_dir / img_path.with_suffix('.txt').name
            if not label_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]

            for line in label_path.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts)
                cls_id = int(cls_id)

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                color = colors[cls_id % len(colors)]
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            out_path = preview_dir / split / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame)
            count += 1

    logger.info("Labeled preview images saved to %s (%s files)", preview_dir, count)
