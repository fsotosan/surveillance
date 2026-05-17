import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VIDEO_ROOT_PATH = os.environ.get('SURVEILLANCE_VIDEO_ROOT', '/Volumes/HDD/video')
MODEL_PATH = os.environ.get('SURVEILLANCE_MODEL_PATH', str(ROOT / 'model' / 'best.pt'))
ACTIVITY_DB_PATH = os.environ.get('SURVEILLANCE_ACTIVITY_DB', str(ROOT / 'db' / 'activity.db'))
TRAINING_DB_PATH = os.environ.get('SURVEILLANCE_TRAINING_DB', str(ROOT / 'db' / 'training.db'))
TRAINING_IMAGES_PATH = os.environ.get('SURVEILLANCE_TRAINING_IMAGES', str(ROOT / 'training' / 'images'))
DEFAULT_MODEL = os.environ.get('SURVEILLANCE_DEFAULT_MODEL', 'yolo11n.pt')
DEVICE = os.environ.get('SURVEILLANCE_DEVICE', '')
MERGE_WINDOW_SECONDS = int(os.environ.get('SURVEILLANCE_MERGE_WINDOW', '300'))
SAMPLING_FPS = int(os.environ.get('SURVEILLANCE_SAMPLING_FPS', '1'))
POLL_INTERVAL = int(os.environ.get('SURVEILLANCE_POLL_INTERVAL', '10'))
ACTIVITY_ROOT_PATH = os.environ.get('SURVEILLANCE_ACTIVITY_ROOT', '/Volumes/HDD/activity')
