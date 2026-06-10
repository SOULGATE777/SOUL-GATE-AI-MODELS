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
| `userId`        | string  | ❌       | User ID for profile assignment (`user_id` also accepted) |
| `frameInterval` | integer | ❌       | Extract every N-th frame (default **1**)              |

**Response:**

```json
{
  "framesExtracted": 24,
  "manifestKey": "face-rotation-dataset/rotacion_horizontal/sess01_horizontal_0_45_manifest.json",
  "sessionId": "sess01",
  "profileId": "00001",
  "rotationType": "horizontal_0_45",
  "perFrame": [
    {
      "frame_id": "00001_y25_p-5_r2",
      "source_video_key": "face-rotation-samples/user42/sess01/HORIZONTAL_0_45.mp4",
      "rotation_type": "HORIZONTAL_0_45",
      "yaw": 22.4,
      "pitch": 1.07,
      "roll": 0.34,
      "quality_score": 0.732,
      "face_detected": true,
      "session_id": "sess01",
      "profile_id": "00001",
      "timestamp_ms": 166.67,
      "s3_frame_key": "face-rotation-dataset/rotacion_horizontal/rh_1D/00001_y25_p-5_r2.jpg"
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
| `frameInterval` | integer | ❌       | Same as `/extract` (default **1**; env `FRAME_INTERVAL`) |

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

**New rotation types** (redesign enums) use the 4-category dataset convention:

```
s3://{bucket}/face-rotation-dataset/{categoria}/{sub_carpeta}/{profileId}_y{yaw}_p{pitch}_r{roll}.jpg
s3://{bucket}/face-rotation-dataset/rotacion_horizontal/{sessionId}_{rotationType}_manifest.json
s3://{bucket}/face-rotation-dataset/profiles_manifest.json
```

**Legacy rotation types** (`frontal`, `left`, `right`, `up`, `down`, `roll`) keep
the previous layout:

```
s3://{bucket}/face-rotation-dataset/{rotationType}/{angleRange}/{sessionId}_{rotationType}_{####}.jpg
```

Example tree (new convention):

```
face-rotation-dataset/
  rotacion_horizontal/
    rh_1D/
      00001_y25_p-5_r2.jpg
    rh_2D/
    rh_1I/
    rh_2I/
    sess01_horizontal_0_45_manifest.json
  rotacion_vertical/
    rv_1A/
    rv_1B/
  rotacion_circular/
    rc_Q1/
    rc_Q2/
    rc_Q3/
    rc_Q4/
  posturas_neutrales/
    pn_frontal/
    pn_perfil_D/
    pn_perfil_I/
  profiles_manifest.json
```

---

## Folder routing (new rotation types)

| `rotationType` (case-insensitive) | Category folder        | Subfolder |
|-----------------------------------|------------------------|-----------|
| `horizontal_0_45`                 | `rotacion_horizontal`  | `rh_1D`   |
| `horizontal_45_90`                | `rotacion_horizontal`  | `rh_2D`   |
| `horizontal_0_neg45`              | `rotacion_horizontal`  | `rh_1I`   |
| `horizontal_neg45_neg90`          | `rotacion_horizontal`  | `rh_2I`   |
| `vertical_0_45`                   | `rotacion_vertical`    | `rv_1A`   |
| `vertical_0_neg45`                | `rotacion_vertical`    | `rv_1B`   |
| `circular`                        | `rotacion_circular`    | `rc_Q1`–`rc_Q4` (yaw quadrant) |

Filenames encode integer Euler degrees: `{profileId}_y{yaw}_p{pitch}_r{roll}.jpg`.

**Pitch convention:** raw solvePnP pitch is normalized with a +180° shift so frontal
neutral reads **0** (range **(-180, 180]**). Positive pitch = looking up; negative =
looking down. Yaw and roll are unchanged.

### Neutral postures (horizontal videos only)

From each horizontal video, the extractor keeps the **best-quality** frame per
neutral category and uploads it under `posturas_neutrales/`:

| Subfolder     | Criteria (approx.)                          |
|---------------|---------------------------------------------|
| `pn_frontal`  | `\|yaw\| < 5°` and `\|pitch\| < 5°`        |
| `pn_perfil_D` | `yaw ≥ 80°` and `\|pitch\| ≤ 15°`          |
| `pn_perfil_I` | `yaw ≤ -80°` and `\|pitch\| ≤ 15°`         |

### Profile IDs

When `userId` / `user_id` is provided (always on `/extract-session`), the service
reads or updates `profiles_manifest.json` and assigns a sequential 5-digit
`profileId` (`00001`, `00002`, …). Without a user ID, frames use `00000`.

---

## Manifest Schema

Per-video manifest: `rotacion_horizontal/{sessionId}_{rotationType}_manifest.json`
(legacy types still use `{rotationType}/{sessionId}_manifest.json`).

Global profile registry: `profiles_manifest.json`:

```json
{
  "profiles": [
    {
      "profile_id": "00001",
      "user_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "sessions": ["550e8400-e29b-41d4-a716-446655440000"]
    }
  ]
}
```

Per-video extraction manifest:

```json
{
  "session_id": "sess01",
  "profile_id": "00001",
  "rotation_type": "horizontal_0_45",
  "source_video_key": "face-rotation-samples/user42/sess01/HORIZONTAL_0_45.mp4",
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
      "frame_id": "00001_y25_p-5_r2",
      "source_video_key": "face-rotation-samples/user42/sess01/HORIZONTAL_0_45.mp4",
      "rotation_type": "horizontal_0_45",
      "yaw": 22.4,
      "pitch": 1.07,
      "roll": 0.34,
      "quality_score": 0.732,
      "face_detected": true,
      "session_id": "sess01",
      "profile_id": "00001",
      "timestamp_ms": 166.67,
      "s3_frame_key": "face-rotation-dataset/rotacion_horizontal/rh_1D/00001_y25_p-5_r2.jpg"
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
| `FRAME_INTERVAL`      | ❌       | `1`            | Default stride for `/extract-session` |

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
# API container must exist first so network api_default is created:
#   cd SOUL-GATE-FRONTEND-WEB/api && docker compose up -d
docker compose up --build -d
```

**Production (EC2):** join the Soul Gate API Docker network so the API can reach
this service by hostname (`FRAME_EXTRACTOR_URL=http://frame_extractor:8020`).
The compose file attaches to external network `api_default`. After updating
compose, recreate the container: `docker compose up -d --force-recreate`.

Verify from the API container:

```bash
docker exec api-api-1 wget -qO- http://frame_extractor:8020/health
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
