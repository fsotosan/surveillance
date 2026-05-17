# Surveillance — Person Activity Tracking from Video

Track people and pets across your home security cameras using YOLO object detection. Designed for camera systems that upload motion-triggered video clips (for example via FTP).

## Video file naming convention

Place videos in `VIDEO_ROOT_PATH/YYYY-MM-DD/` with the format:

```
{RoomName}_{CameraNumber}_{HubName}_{YYYYMMDDhhmmss}.mp4
```

Example:

```
/Volumes/HDD/video/2026-05-12/Cocina_02_Hub Abuelita_20260512013029.mp4
```

## Installation

```bash
git clone <repo>
cd surveillance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place your trained YOLO model at `model/best.pt`.

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SURVEILLANCE_VIDEO_ROOT` | `/Volumes/HDD/video` | Root folder containing `YYYY-MM-DD` subfolders |
| `SURVEILLANCE_MODEL_PATH` | `model/best.pt` | Path to YOLO model weights |
| `SURVEILLANCE_ACTIVITY_DB` | `db/activity.db` | SQLite database for activity records |
| `SURVEILLANCE_TRAINING_DB` | `db/training.db` | SQLite database for onboarded training data |
| `SURVEILLANCE_TRAINING_IMAGES` | `training/images` | Directory to store onboarded frame images |
| `SURVEILLANCE_DEFAULT_MODEL` | `yolo11n.pt` | Base model for training (downloaded on demand) |
| `SURVEILLANCE_DEVICE` | auto | Torch device (`mps`, `cuda`, `cpu`, or empty for auto) |
| `SURVEILLANCE_MERGE_WINDOW` | `300` | Seconds to merge consecutive detections of the same class/room |
| `SURVEILLANCE_SAMPLING_FPS` | `1` | Frames per second to sample from video |
| `SURVEILLANCE_POLL_INTERVAL` | `10` | Seconds between folder polls in surveillance mode |
| `SURVEILLANCE_ACTIVITY_ROOT` | `activity_images` | Root folder for activity snapshot images |

## Usage

Global flags:

| Flag | Description |
|------|-------------|
| `-v` / `--verbose` | Enable debug logging |

### `surveillance` — continuous monitoring

```bash
python -m surveillance surveillance [--init] [--bulk]
```

Monitors the latest date folder under `VIDEO_ROOT_PATH`. For each new video:
1. Runs YOLO prediction on sampled frames
2. Detects named people (e.g. "dad", "mom", "grandma") or falls back to generic classes
3. Merges consecutive same-room detections into single activity entries
4. Saves labeled snapshot images (bounding-box visualizations) of the first and last detection frame per activity to `SURVEILLANCE_ACTIVITY_ROOT/YYYY-MM-DD/`

`--init` — if no trained model exists at `model/best.pt`, auto-downloads and uses the base model (`yolo11n.pt`).

`--bulk` — process all video folders under `VIDEO_ROOT_PATH` from oldest to newest, then exit. Useful for initial backfill of historical video data.

### `onboard` — generate training data

```bash
python -m surveillance onboard [--init] <video.mp4> <classname> <username>
```

Example — label a video of dad walking through the kitchen:

```bash
python -m surveillance onboard --init dad_clip.mp4 person dad
```

For each frame where the model detects `<classname>`, the frame is saved as a JPEG and its annotation (in YOLO format) is stored in the training database. The frame timestamp is derived from the video file's modification time.

`--init` — same fallback behavior as surveillance.

### `train` — train a new YOLO model

```bash
python -m surveillance train [--epochs 100] [--imgsz 640] [--batch 16] [--export-only]
```

Exports all onboarded images from the training database into a YOLO dataset (80/20 train/val split), backs up `model/best.pt`, trains a new model, and saves it to `model/best.pt`.

`--export-only` — export the dataset and generate labeled preview images (bounding-box visualizations in `training/dataset/labeled_preview/`) without training. The dataset directory is left in place for review.

#### Training in Google Colab (recommended for GPU)

For faster training on a GPU, use the provided `train.ipynb` notebook in Google Colab:

1. Upload `train.ipynb` to Google Drive or open it directly via Colab
2. Mount your Drive and edit the configuration cell at the top:
   - Set `REPO_URL` to your fork (if you forked the repo)
   - Set `PARENT_DIR` to where the repo should live in Drive
   - Point `TRAINING_IMAGES` and `TRAINING_DB` to your onboarded data
3. Run all cells — the notebook clones/updates the repo, installs dependencies, and runs training

The notebook uses `git pull` on subsequent runs, so you can re-run simply by updating the dataset and executing all cells again.

### `report` — query activity

```bash
python -m surveillance report "<from>" "<to>" [--class FILTER]
```

Example:

```bash
python -m surveillance report "2026-05-12 00:00:00" "2026-05-12 23:59:59" --class grandma
```

### `find` — last-seen lookup

```bash
python -m surveillance find <classname>
```

Example:

```bash
python -m surveillance find dad
# dad: last seen in Cocina at 2026-05-12 08:30:29
```

## Detection logic

The system distinguishes between **generic classes** (`person`, `man`, `woman`, `dog`, `cat`) and **named classes** (any other name your custom model is trained to detect). When a frame contains detections, named classes are preferred — if any named class is found, generic detections are discarded. This avoids recording duplicate "person" entries when a custom model already identifies the specific person.

## Activity merge logic

When a person is detected, the system checks for an existing activity entry with the same (class, room, camera) whose `datetime_to` is within the merge window (default 5 minutes). If found, the entry is extended. Otherwise a new entry is created.

Examples:
- Grandma in living room 3–5 pm → single entry
- Grandma moves to kitchen at 5 pm → new kitchen entry
- Grandma returns to living room at 5:10 pm → new living room entry (gap > 5 min from previous)

## Preview images (export-only)

When training with `--export-only`, the system generates labeled preview images showing bounding boxes with class names. These are written to `training/dataset/labeled_preview/` and are useful for verifying dataset correctness before committing to a full training run.
