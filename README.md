# Steel Sizing — Billet Dimension Measurement System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-red) ![Pose_Estimation](https://img.shields.io/badge/YOLOv8-Pose-orange) ![Redis](https://img.shields.io/badge/Redis-Queue-red) ![Docker](https://img.shields.io/badge/Docker-Compose-blue)

Automated billet dimension measurement system using AI. Combines YOLOv8 object detection with pose estimation keypoints to measure the width, height, and length of steel billets on the production line without physical contact.

## Features

- **Non-contact measurement** — camera-based dimensional analysis replaces manual gauging
- **Dual AI pipeline** — YOLOv8 detection (`classdet`) locates billets; YOLOv8 pose (`classpose`) extracts keypoints for dimension calculation
- **Combined mode** — `classdetpose` runs both models in a unified pipeline
- **Redis task queue** — async worker processes measurement tasks from a Redis queue
- **ROI selector** — visual tool for defining the measurement region of interest
- **Pixel-to-mm mapping** — configurable scale mapping (pixels → real-world units)
- **FastAPI REST** — submit measurement jobs and query results
- **Database logging** — measurement results (width, height, length) stored per billet

## Tech Stack

| Component | Technology |
|---|---|
| Object Detector | YOLOv8 Detection (`det/best.pt`) |
| Dimension Extractor | YOLOv8 Pose (`pose/best.pt`) |
| API Server | FastAPI + Uvicorn |
| Task Queue | Redis |
| Worker | Python background worker |
| Containerization | Docker Compose |

## Architecture

```
Camera / RTSP Stream / Image Upload
            │
            ▼
    FastAPI REST API
    POST /measure  →  Redis Task Queue
            │
            ▼
       Background Worker
    ┌────────────────────────┐
    │  YOLOv8 Detector       │  ← locates billet ROI
    │  YOLOv8 Pose Estimator │  ← extracts 4 corner keypoints
    │  Dimension Calculator  │  ← pixel → mm via mapping
    └────────────────────────┘
            │
            ▼
    Result stored → Database
    GET /result/{task_id}  →  { width_mm, height_mm, length_mm }
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

# 3. Configure pixel-to-mm mapping
python utils/roi_selector.py --camera rtsp://username:password@192.168.1.100:554/

# 4. Start services
docker compose up -d --build
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/measure` | Submit image or stream for measurement |
| `GET` | `/result/{task_id}` | Get measurement result by task ID |
| `GET` | `/health` | Service and Redis health check |
| `GET` | `/history` | List recent measurement results |

### Example: Measure from Image Upload

```bash
curl -X POST http://localhost:9000/measure \
  -F "file=@/path/to/billet_frame.jpg" \
  -F "task_type=both"
```

**Response:**

```json
{
  "task_id": "f3a4b2c1-...",
  "status": "queued"
}
```

### Example: Get Result

```bash
curl http://localhost:9000/result/f3a4b2c1-...
```

```json
{
  "task_id": "f3a4b2c1-...",
  "status": "completed",
  "width_mm": 130.4,
  "height_mm": 128.9,
  "length_mm": 6012.0,
  "confidence": 0.92,
  "timestamp": "2024-01-15T12:34:56"
}
```

## ROI & Scale Configuration

```bash
# Select measurement ROI interactively
python utils/roi_selector.py --camera rtsp://...

# Configure pixel-to-mm scale mapping
python utils/mapping.py --reference_width_mm 130 --reference_pixels 312
```

## Worker

The background worker processes tasks from the Redis queue independently:

```bash
# Worker runs automatically in Docker Compose
# To run manually for testing:
python worker/worker.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
