# Coding Agent Instructions

## Project Overview

Surveillance — a Python CLI tool that tracks people/activity from security camera videos using YOLO object detection.

## Architecture

```
surveillance/
├── __init__.py         # Version
├── __main__.py         # Entry: `python -m surveillance`
├── cli.py              # argparse subcommands
├── config.py           # Env-based configuration with defaults
├── db.py               # ActivityDB & TrainingDB (SQLite)
├── video.py            # Filename parsing, frame sampling
├── yolo_utils.py       # YOLO model wrapper, Detection dataclass
├── surveil.py          # `surveillance` action — continuous monitor
├── onboard.py          # `onboard` action — training data generation
├── train_action.py     # `train` action — YOLO training
├── report.py           # `report` action — query activity
└── find_action.py      # `find` action — last-seen lookup
```

## Key Conventions

- **Configuration**: all settings come from environment variables with `SURVEILLANCE_` prefix (see `config.py` for defaults). Never hardcode paths.
- **Database**: ActivityDB for activity tracking (+ processed_files table). TrainingDB for onboarded image annotations. Both use `sqlite3.Row` for named access.
- **Filename parsing**: `RoomName_CameraNumber_HubName_YYYYMMDDhhmmss.mp4` — the 14-digit timestamp anchors parsing; HubName may contain underscores.
- **Activity merging**: `upsert_activity()` checks the most recent entry for the same (class, room, camera). If its `datetime_to` is within `MERGE_WINDOW_SECONDS` of the new event, it extends the entry; otherwise it inserts a new row.
- **Class priority**: `get_relevant_classes()` prefers non-generic classes (e.g. "dad") over generic ones (`person`, `man`, `woman`, `dog`, `cat`).
- **Onboard annotation format**: Each annotation line is `{username} {x_center} {y_center} {width} {height}` (normalized YOLO format). During training export, `username` is replaced with the numeric class ID.
- **Onboard frame timestamp**: Derived from `os.path.getmtime()` of the video file, not from filename parsing.
- **Training export**: Images copied to `training/dataset/` with 80/20 train/val split. `data.yaml` written with absolute paths. Old `model/best.pt` is timestamp-backupped before training. With `--export-only`, labeled preview images (bounding-box visualizations) are generated in `training/dataset/labeled_preview/`.
- **Video sampling**: 1 fps by default (`SURVEILLANCE_SAMPLING_FPS`). Frames are processed one at a time via `cv2.VideoCapture` + `model(frame)`.
- **Logging**: Use `logging.getLogger(__name__)` in every module. Status/progress/errors go to `logger.*()` (→ stderr). Command output (report table, find result) stays as `print()` (→ stdout). Set `-v` for debug output — this calls `setup_logging()` from `__init__.py` which also quiets `ultralytics` to WARNING.
- **Error handling**: All errors in the surveillance loop are caught and logged; the loop continues polling. `KeyboardInterrupt` is also caught for clean shutdown.
- **File stability**: Before processing a new video file, the system waits for its size to stabilise (2 consecutive identical reads).
- **Processed file tracking**: Stored in `activity.db` `processed_files` table for persistence across restarts.
- **`--init` flag**: Both `surveillance` and `onboard` commands support `--init`. When set, if no trained model exists at `model/best.pt`, the system falls back to `SURVEILLANCE_DEFAULT_MODEL` (e.g. `yolo11n.pt`), downloading it if needed.
- **Device resolution**: `yolo_utils._resolve_device()` selects the best available device — MPS (Apple Silicon) > CUDA > CPU — with logging. The `SURVEILLANCE_DEVICE` env var overrides auto-detection.

## Development Rules

1. When adding new features, always check `config.py` first for existing environment variable patterns.
2. The `GENERIC_CLASSES` set in `surveil.py` controls which YOLO class names are considered "generic". When training a model with custom people, those names should NOT be in this set.
3. Database migrations should use `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` patterns — no migration framework.
4. The `model/` directory contains `best.pt` (the current production model). Backups use the naming convention `best_YYYYMMDDHHmmss.pt`.
5. The `run_train()` function copies images into a temporary `training/dataset/` directory. With `--export-only` the export happens but training is skipped, leaving the dataset in place for review. Otherwise the directory is deleted after training completes.
6. When `--export-only` is used, `_generate_labeled_preview()` creates bounding-box visualizations in `training/dataset/labeled_preview/` for dataset verification.
7. The surveillance loop in `run_surveillance()` always catches exceptions (including `KeyboardInterrupt`) to ensure clean database connection shutdown.
