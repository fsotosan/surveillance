import re
from datetime import datetime, timedelta


def parse_video_filename(filename: str):
    name = filename
    if name.endswith('.mp4'):
        name = name[:-4]

    match = re.search(r'_(\d{14})$', name)
    if not match:
        return None

    timestamp_str = match.group(1)
    prefix = name[:match.start()]
    parts = prefix.split('_')

    if len(parts) < 3:
        return None

    room_name = parts[0]
    camera_number = parts[1]
    hub_name = '_'.join(parts[2:])

    try:
        start_dt = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    except ValueError:
        return None

    return room_name, camera_number, hub_name, start_dt


def sample_frames(video_path, sampling_fps: float = 1, start_dt: datetime | None = None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30

    sample_interval = max(1, int(round(video_fps / max(1, sampling_fps))))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            ts = frame_idx / video_fps
            if start_dt is not None:
                frame_dt = start_dt + timedelta(seconds=ts)
            else:
                frame_dt = datetime.now()
            yield frame_dt, frame

        frame_idx += 1

    cap.release()
