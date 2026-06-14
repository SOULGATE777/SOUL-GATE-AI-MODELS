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
| `dedup`         | boolean | ❌       | Keep only the best frame per pose bin (default env `FRAME_DEDUP_ENABLED`, **true**) |
| `poseBinStep`   | integer | ❌       | Angle bin width in degrees for dedup (default env `POSE_BIN_STEP_DEG`, **2**) |
| `minQualityScore` | number | ❌      | Drop frames below this quality score (default env `MIN_QUALITY_SCORE`, **0.0**) |

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
| `captureSource` | string  | ❌       | `web` → filename prefix `W`; `mobile` → `M` (see dataset naming doc) |
| `dedup`         | boolean | ❌       | Same as `/extract` (default env `FRAME_DEDUP_ENABLED`, **true**) |
| `poseBinStep`   | integer | ❌       | Same as `/extract` (default env `POSE_BIN_STEP_DEG`, **2**) |
| `minQualityScore` | number | ❌      | Same as `/extract` (default env `MIN_QUALITY_SCORE`, **0.0**) |

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
s3://{bucket}/face-rotation-dataset/{categoria}/{sub_carpeta}/{W|M}{profileId}_y{yaw}_p{pitch}_r{roll}.jpg
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

Subfolders are chosen from **measured Euler angles** at extraction time, not from
the source video `rotationType` label alone. Combined capture blobs (e.g.
`HORIZONTAL_0_45` and `HORIZONTAL_45_90` share one 0°→90° sweep) are split into
the correct sub-range folders automatically.

| Category folder       | Subfolder | Routing rule (measured angle) |
|-----------------------|-----------|--------------------------------|
| `rotacion_horizontal` | `rh_1D`   | yaw in `[0°, 45°)`             |
| `rotacion_horizontal` | `rh_2D`   | yaw `≥ 45°`                    |
| `rotacion_horizontal` | `rh_1I`   | yaw in `[-45°, 0°)`            |
| `rotacion_horizontal` | `rh_2I`   | yaw `< -45°`                   |
| `rotacion_vertical`   | `rv_1A`   | pitch `≥ 0°`                   |
| `rotacion_vertical`   | `rv_1B`   | pitch `< 0°`                   |
| `rotacion_circular`   | `rc_Q1`–`rc_Q4` | yaw quadrant (unchanged) |

The `rotationType` on the source MP4 still selects the **category folder**
(`rotacion_horizontal`, etc.) via `CATEGORY_MAP`; only the subfolder band uses
live yaw/pitch from MediaPipe + solvePnP.

Filenames encode integer Euler degrees: `{profileId}_y{yaw}_p{pitch}_r{roll}.jpg`.

**Pitch convention:** raw solvePnP pitch is normalized with a +180° shift so frontal
neutral reads **0** (range **(-180, 180]**). Positive pitch = looking up; negative =
looking down. Yaw and roll are unchanged.

### Video orientation handling

Phone clips may decode sideways when OpenCV reads the MP4 without rotation
metadata. Before frame extraction, the service samples ~15 evenly spaced frames,
tries rotations `{0°, 90°, 180°, 270°}`, and picks the orientation with the most
face detections (ties → no rotation). Every extracted frame is rotated to that
upright orientation before Euler estimation, so angles and subfolder routing stay
consistent with the table above.

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
    "quality_score_mean": 0.741,
    "selection": {
      "dedup_enabled": true,
      "pose_bin_step_deg": 2,
      "min_quality_score": 0.0,
      "frames_processed": 291,
      "frames_selected": 34,
      "frames_skipped_no_face": 6,
      "frames_skipped_low_quality": 0,
      "neutral_frames": 1
    }
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
| `FRAME_DEDUP_ENABLED` | ❌       | `true`         | Keep only the best frame per pose bin (set `false` for legacy keep-all) |
| `POSE_BIN_STEP_DEG`   | ❌       | `2` (library) / `6` (compose) | Pose-bin width in degrees; raise to thin the dataset further |
| `MIN_QUALITY_SCORE`   | ❌       | `0.0`          | Drop frames below this composite quality score |

### Pose-bin deduplication

The extractor samples every frame (`FRAME_INTERVAL=1`) so fast sweeps never miss
an angle, but slow movement produces bursts of near-identical poses (and roll
jitter) that inflate the dataset with look-alikes. With `FRAME_DEDUP_ENABLED=true`
(default), detected frames are grouped into pose bins on the meaningful axes —
**yaw** for horizontal, **pitch** for vertical, **yaw+pitch** for circular — and
only the highest `quality_score` frame per bin is uploaded. Frames without a face
or below `MIN_QUALITY_SCORE` are dropped. Roll is excluded from binning (it is the
main jitter source). Neutral-posture selection is unchanged. Set
`FRAME_DEDUP_ENABLED=false` to restore the legacy keep-every-frame behaviour.

**Tuning `POSE_BIN_STEP_DEG`.** The bin width is the main lever for visual
redundancy: frames closer than the step look identical, so a small step (e.g.
`2`) keeps many near-duplicates. `docker-compose.yml` defaults to `6` for a
visibly thinner dataset. Note that **circular** uses a 2-D (yaw+pitch) bin, so it
keeps more frames than the 1-D rotations at the same step — raise the step to `8`
or `10` if circular still looks too dense. Override per environment with
`POSE_BIN_STEP_DEG=8 docker compose up -d` (no rebuild needed).

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
