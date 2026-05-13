import logging

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float


def _resolve_device():
    import torch
    from surveillance import config

    if config.DEVICE:
        return config.DEVICE

    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal GPU)")
        return "mps"
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return "cuda"
    logger.info("Using CPU")
    return "cpu"


def load_model(model_path: str):
    from surveillance import config
    from ultralytics import YOLO

    device = _resolve_device()
    logger.info("Loading model on device: %s", device)
    model = YOLO(str(model_path))
    model.to(device)
    return model


def predict_frame(model, frame):
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            h, w = frame.shape[:2]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            detections.append(Detection(
                class_name=class_name,
                confidence=conf,
                x_center=x_center,
                y_center=y_center,
                width=bw,
                height=bh,
            ))
    return detections
