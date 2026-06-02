# Frame Extractor — ML Dataset Pipeline Service

> **Phase 4 — Face Rotation Dataset Module**

Standalone FastAPI microservice that reads face-rotation videos from AWS S3,
extracts frames with OpenCV, annotates each frame with Euler angles
(yaw / pitch / roll) via MediaPipe Face Mesh, computes a quality score, and
writes the results (JPEG frames + JSON manifest) back to S3.

---

## Port

**8020** — verified free across all `docker-compose.yml` files in the monorepo.

---

## Endpoints

### `GET /health`

Health check. Always returns 200.

```json
{ "status": "ok", "service": "frame_extractor" }
```

---

### `POST /extract`

Process a single video.

**Request body (JSON, camelCase):**

| Field           | Type    | Required | Description                                           |
|-----------------|---------|----------|-------------------------------------------------------|
| `s3Key`         | string  | ✅       | Source S3 key, e.g. `face-rotation-samples/u/s/frontal.mp4` |
| `s3Bucket`      | string  | ❌       | S3 bucket (falls back to `AWS_BUCKET_NAME`)           |
| `rotationType`  | string  | ✅       | New: `horizontal_0_45 \| horizontal_45_90 \| horizontal_0_neg45 \| horizontal_neg45_neg90 \| vertical_0_45 \| vertical_0_neg45 \| circular`. Legacy still accepted: `frontal \| left \| right \| up \| down \| roll` |
| `sessionId`     | string  | ✅       | Unique session identifier                             |
| `frameInterval` | integer | ❌       | Extract every N-th frame (default **5**)              |

**Response:**

```json
{
  "framesExtracted": 24,
  "manifestKey": "face-rotation-dataset/horizontal_0_45/sess01_manifest.json",
  "sessionId": "sess01",
  "rotationType": "horizontal_0_45",
  "perFrame": [
    {
      "frame_id": "sess01_horizontal_0_45_0000",
      "source_video_key": "face-rotation-samples/user42/sess01/horizontal_0_45.mp4",
      "rotation_type": "horizontal_0_45",
      "yaw": 22.4,
      "pitch": 1.07,
      "roll": 0.34,
      "quality_score": 0.732,
      "face_detected": true,
      "session_id": "sess01",
      "timestamp_ms": 166.67,
      "s3_frame_key": "face-rotation-dataset/horizontal_0_45/yaw_20_25/sess01_horizontal_0_45_0000.jpg"
    }
  ]
}
```

---

### `POST /extract-session`

Convenience — processes all rotation types for a session.

**Request body:**

| Field           | Type     | Required | Description                                     |
|-----------------|----------|----------|-------------------------------------------------|
| `userId`        | string   | ✅       | User identifier                                 |
| `sessionId`     | string   | ✅       | Session identifier                              |
| `rotationTypes` | string[] | ❌       | Defaults to the 7 redesign types: `["horizontal_0_45", "horizontal_45_90", "horizontal_0_neg45", "horizontal_neg45_neg90", "vertical_0_45", "vertical_0_neg45", "circular"]` |

Constructs source keys automatically:

```
face-rotation-samples/{userId}/{sessionId}/{rotationType}.mp4
```

**Response:**

```json
{
  "sessionId": "sess01",
  "results": [ ... ],
  "errors": []
}
```

---

## S3 Key Layout

### Input (read-only)

```
s3://{bucket}/face-rotation-samples/{userId}/{sessionId}/{rotationType}.mp4
```

### Output (written by this service)

```
s3://{bucket}/face-rotation-dataset/{rotationType}/{angleRange}/{frameId}.jpg
s3://{bucket}/face-rotation-dataset/{rotationType}/{sessionId}_manifest.json
```

Example tree:

```
face-rotation-dataset/
  horizontal_0_45/
    yaw_0_5/
      sess01_horizontal_0_45_0000.jpg
    yaw_40_45/
      sess01_horizontal_0_45_0012.jpg
    sess01_manifest.json
  vertical_0_45/
    pitch_30_35/
      sess01_vertical_0_45_0007.jpg
    sess01_manifest.json
  circular/
    yaw_0_90/
      sess01_circular_0000.jpg
    yaw_-90_0/
      sess01_circular_0005.jpg
    sess01_manifest.json
```

---

## Angle-Range Bucketing Scheme

Each extracted frame is assigned to a bucket based on the **dominant Euler
angle** for its rotation type.  Non-circular types use **5-degree wide bins**;
the `circular` type uses **90-degree yaw quadrants** (see below).

| Rotation type             | Dominant axis | Bucket label format |
|---------------------------|---------------|---------------------|
| `horizontal_0_45`         | **yaw**       | `yaw_{lo}_{hi}`     |
| `horizontal_45_90`        | **yaw**       | `yaw_{lo}_{hi}`     |
| `horizontal_0_neg45`      | **yaw**       | `yaw_{lo}_{hi}`     |
| `horizontal_neg45_neg90`  | **yaw**       | `yaw_{lo}_{hi}`     |
| `vertical_0_45`           | **pitch**     | `pitch_{lo}_{hi}`   |
| `vertical_0_neg45`        | **pitch**     | `pitch_{lo}_{hi}`   |
| `circular`                | **yaw**       | `yaw_{lo}_{hi}` (90° quadrants) |
| `frontal` *(legacy)*      | **yaw**       | `yaw_{lo}_{hi}`     |
| `left` / `right` *(legacy)* | **yaw**     | `yaw_{lo}_{hi}`     |
| `up` / `down` *(legacy)*  | **pitch**     | `pitch_{lo}_{hi}`   |
| `roll` *(legacy)*         | **roll**      | `roll_{lo}_{hi}`    |
| *(other)*                 | yaw (fallback)| `yaw_{lo}_{hi}`     |
| no face                   | —             | `no_face`           |

For non-circular types, **`lo`** and **`hi`** are integer multiples of 5.

Examples:

| Yaw (°) | Bucket label  |
|---------|---------------|
| −7.3    | `yaw_-10_-5`  |
| 0.0     | `yaw_0_5`     |
| 12.8    | `yaw_10_15`   |
| −22.1   | `yaw_-25_-20` |

### Circular Quadrant Splitting

The `circular` phase is a single continuous head rotation. After computing yaw
per frame, frames are split into **four signed-yaw quadrants** (90-degree bins),
staying consistent with the `{axis}_{lo}_{hi}` label format:

| Quadrant | Yaw range        | Bucket label   |
|----------|------------------|----------------|
| Q1       | `0°` to `90°`    | `yaw_0_90`     |
| Q2       | `90°` to `180°`  | `yaw_90_180`   |
| Q3       | `-180°` to `-90°`| `yaw_-180_-90` |
| Q4       | `-90°` to `0°`   | `yaw_-90_0`    |

Frames where no face is detected fall into `no_face`. In practice MediaPipe
loses the face near profile (±90°), so most usable circular frames land in
`yaw_0_90` (Q1) and `yaw_-90_0` (Q4).

---

## Manifest Schema

`{sessionId}_manifest.json` — uploaded alongside frames:

```json
{
  "session_id": "sess01",
  "rotation_type": "horizontal_0_45",
  "source_video_key": "face-rotation-samples/user42/sess01/horizontal_0_45.mp4",
  "created_at": 1706890000.0,
  "summary": {
    "session_id": "sess01",
    "rotation_type": "horizontal_0_45",
    "total_frames": 24,
    "frames_with_face": 22,
    "angle_min": 1.2,
    "angle_max": 44.8,
    "quality_score_mean": 0.741
  },
  "frames": [
    {
      "frame_id": "sess01_horizontal_0_45_0000",
      "source_video_key": "face-rotation-samples/user42/sess01/horizontal_0_45.mp4",
      "rotation_type": "horizontal_0_45",
      "yaw": 22.4,
      "pitch": 1.07,
      "roll": 0.34,
      "quality_score": 0.732,
      "face_detected": true,
      "session_id": "sess01",
      "timestamp_ms": 166.67,
      "s3_frame_key": "face-rotation-dataset/horizontal_0_45/yaw_20_25/sess01_horizontal_0_45_0000.jpg"
    }
  ]
}
```

**Quality score sub-components (all normalised to [0, 1]):**

| Component             | Weight | Description                              |
|-----------------------|--------|------------------------------------------|
| Sharpness (Laplacian) | 40 %   | `min(laplacian_var / 200, 1.0)`          |
| Face size             | 30 %   | Face bbox area as fraction of frame      |
| Detection confidence  | 30 %   | MediaPipe detection confidence           |

---

## Environment Variables

| Variable              | Required | Default        | Description                  |
|-----------------------|----------|----------------|------------------------------|
| `AWS_REGION`          | ❌       | `us-east-1`    | AWS region                   |
| `AWS_ACCESS_KEY_ID`   | ✅       | —              | IAM access key               |
| `AWS_SECRET_ACCESS_KEY` | ✅     | —              | IAM secret key               |
| `AWS_BUCKET_NAME`     | ✅       | —              | Default S3 bucket            |

---

## Running Locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_BUCKET_NAME=my-soul-ai-bucket

# 4. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8020 --reload
```

Open the interactive docs at <http://localhost:8020/docs>.

---

## Running with Docker

```bash
# Build
docker build -t soul-ai/frame_extractor:latest .

# Run
docker run -p 8020:8020 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_BUCKET_NAME=my-soul-ai-bucket \
  soul-ai/frame_extractor:latest
```

Or with Docker Compose:

```bash
cp .env.example .env   # fill in AWS credentials
docker-compose up --build
```

---

## Architecture Notes

- **Read-only for source data**: the service only reads from
  `face-rotation-samples/` and never modifies it.
- **No database**: no connections to PostgreSQL or any other DB.
- **No Node/API gateway interaction**: standalone HTTP microservice.
- **Lazy model loading**: `FaceEulerEstimator` is created on first request
  (mirrors the `MultiModelLoader` pattern used across `frontal_prod/` services).
- **Temp-file cleanup**: the downloaded video is always removed in a `finally`
  block, even when frame extraction or S3 upload fails.
