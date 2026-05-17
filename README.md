# Steel Sizing — Billet Dimension Measurement System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-red) ![Pose_Estimation](https://img.shields.io/badge/YOLOv8-Pose-orange) ![Redis](https://img.shields.io/badge/Redis-Queue-red) ![Docker](https://img.shields.io/badge/Docker-Compose-blue)

Automated billet dimension measurement system using AI. Combines YOLOv8 object detection with pose estimation keypoints to measure the width, height, and length of steel billets on the production line without physical contact.

## Features

- **Non-contact measurement** — camera-based dimensional analysis replaces manual gauging
- **Dual AI pipeline** — YOLOv8 detection (`BilletDetectionApp`) locates billets; YOLOv8 pose (`BilletSizingApp`) extracts keypoints for length calculation
- **Three task modes** — `det` (detection only), `pose` (length via keypoints), `both` (full pipeline)
- **Dual-thread worker** — API task worker (Redis BLPOP) + camera stream worker (Redis Streams XREADGROUP) run concurrently
- **ROI + line_x threshold** — only billets whose center crosses the configured vertical reference line are measured
- **Range-based pixel-to-mm mapping** — configurable YAML lookup tables map pixel ranges to real-world dimensions
- **FastAPI REST** — submit jobs by file upload or source path; auto-detects image vs. video by extension
- **SQLite logging** — measurement results (width_mm, height_mm, length_mm) stored per billet with timestamp

## Tech Stack

| Component | Technology |
|---|---|
| Billet Detector | YOLOv8 Detection (`models/det/best.pt`) |
| Dimension Extractor | YOLOv8 Pose (`models/pose/best.pt`) |
| API Server | FastAPI + Uvicorn |
| Task Queue | Redis list (`billet_tasks`) |
| Camera Stream | Redis Streams (`billet_camera_stream`) |
| Database | SQLite (`billet_data.db`) |
| Containerization | Docker Compose |

## Architecture

```
Camera / RTSP Stream / Image Upload
            │
            ▼
    FastAPI REST API
    POST /process_upload  →  Redis List (billet_tasks)
    POST /process_source  →  Redis List (billet_tasks)
            │
            ▼
       Background Worker (two concurrent threads)
    ┌───────────────────────────────────────────┐
    │  Thread 1: API Task Worker                │
    │    BLPOP billet_tasks                     │
    │    → image / video / RTSP processing      │
    │                                           │
    │  Thread 2: Camera Stream Worker           │
    │    XREADGROUP billet_camera_stream        │
    │    → live camera frame processing         │
    └───────────────────────────────────────────┘
            │ (both threads run)
            ▼
    ┌────────────────────────────────────────┐
    │  ROI filter + line_x threshold         │
    │  YOLOv8 Detector → bounding boxes      │
    │  YOLOv8 Pose → 2 endpoint keypoints    │
    │  Euclidean distance → length_px        │
    │  Pixel-to-mm lookup table → length_mm  │
    └────────────────────────────────────────┘
            │
            ▼
    SQLite billet_data.db
    (timestamp, width_mm, height_mm, length_mm)
```

## Prerequisites

- Docker & Docker Compose
- YOLOv8 detection weights at `models/det/best.pt`
- YOLOv8 pose weights at `models/pose/best.pt`
- Redis instance (included in Docker Compose)

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/sadra-ai25/steel-sizing.git
cd steel-sizing

# 2. Place model weights
mkdir -p models/det models/pose
cp /path/to/det_best.pt  models/det/best.pt
cp /path/to/pose_best.pt models/pose/best.pt

# 3. Configure ROI and pixel mapping
python utils/roi_selector.py          # select ROI interactively
# Then edit config/pixel_mapping.yaml with your known dimension ranges

# 4. Start services
docker compose up -d --build
```

## Configuration

### `config/config.yaml` — ROI and measurement line

```yaml
roi:
  x_min: 0
  y_min: 0
  x_max: 3840
  y_max: 2160
line_x: 1920   # vertical threshold line; billets must cross this to be measured
```

### `config/pixel_mapping.yaml` — pixel-range to mm lookup table

```yaml
dimensions:
  "130x130":
    min: 290
    max: 340
  "150x150":
    min: 340
    max: 390
length:
  "6000":
    min: 2800
    max: 3100
  "12000":
    min: 5600
    max: 6200
```

Billet width/height in pixels are matched against dimension ranges; keypoint Euclidean distance is matched against length ranges. Unmatched values fall back to raw pixel values.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/process_upload` | Upload image or video file for measurement |
| `POST` | `/process_source` | Submit by file path or RTSP URL |
| `GET` | `/` | Service health check |

### Example: Upload Image

```bash
curl -X POST http://localhost:9000/process_upload \
  -F "file=@/path/to/billet_frame.jpg" \
  -F "task_type=both"
```

**Response:**

```json
{
  "status": "queued",
  "task_id": "f3a4b2c1-...",
  "detected_type": "image",
  "message": "File uploaded as image and queued."
}
```

### Example: Submit by Source Path

```bash
curl -X POST http://localhost:9000/process_source \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "rtsp",
    "source_path": "rtsp://camera-host:554/stream",
    "task_type": "both"
  }'
```

**Response:**

```json
{
  "status": "queued",
  "message": "Task f3a4b2c1-... queued."
}
```

## Task Types

| `task_type` | Models Used | Output |
|---|---|---|
| `det` | YOLOv8 Detection only | Bounding boxes + width/height in mm |
| `pose` | YOLOv8 Pose only | Keypoints + length in mm |
| `both` | Detection + Pose | Full dimensional measurement |

## Worker

The background worker runs two concurrent threads automatically inside Docker Compose:

```bash
# To run manually for testing:
python worker/worker.py
```

- **API Task Worker** — blocks on `BLPOP billet_tasks` and processes image, video, or RTSP tasks queued by the API
- **Camera Stream Worker** — reads from Redis Stream `billet_camera_stream` via `XREADGROUP`, processes live camera frames pushed by a separate producer

## Pixel-to-mm Mapping

Cross-section dimensions (width × height) and billet length are resolved through a YAML-based range lookup, not a linear scale factor. Each known billet size maps to a pixel count range:

```
detected_width_px=310 → matches "130x130" (range 290–340) → width_mm = 130
keypoint_distance_px=2950 → matches "6000" (range 2800–3100) → length_mm = 6000
```

If no range matches, the raw pixel value is stored as a fallback.

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
