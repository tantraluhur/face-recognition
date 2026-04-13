# Face Recognition / Comparison

A browser-based face comparison app built with Next.js, TypeScript, Tailwind CSS, and ONNX Runtime. Upload a profile photo, open the webcam, and see real-time match results.

The project also ships an **optional Python backend** (FastAPI + InsightFace) that runs bigger models so you can compare accuracy side-by-side.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Performance](#performance)
8. [Troubleshooting](#troubleshooting)
9. [Frontend vs Backend](#frontend-vs-backend)
10. [Knowledge Base](#knowledge-base) — deep dive into models, benchmarks, and concepts
11. [Glossary](#glossary)

---

## Quick Start

### Frontend

```bash
yarn install
yarn dev
```

Open http://localhost:3000. The frontend works standalone — backend cards show "Backend offline" until you start the Python server.

### Backend (optional, for accuracy comparison)

```bash
cd python
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**On first run**, InsightFace downloads ~450 MB of model files to `~/.insightface/models/`. `buffalo_l` loads eagerly at startup; `antelopev2` loads lazily on the first `/api/verify` request. You should see:

```
Starting backend...
Loading buffalo_l (det_size=(640, 640), ctx=cpu)...
buffalo_l ready in 12.3s
Backend ready. Docs: http://localhost:8000/docs
```

Refresh the frontend — the badge at the top flips to "Backend: online" and the right two cards start showing scores.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          BROWSER                                  │
│                                                                   │
│   1. Detect face with SCRFD-500M       (frontend, 2.4 MB)         │
│   2. Align face with 5 landmarks       (frontend, JS)             │
│   3. Embed with MobileFaceNet          (frontend, 13 MB)          │
│   4. Snapshot webcam frame as JPEG                                │
│   5. Send both images to Python backend ──────────────────┐       │
│                                                           │       │
└───────────────────────────────────────────────────────────│───────┘
                                                            │
                                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PYTHON BACKEND (optional)                      │
│                    FastAPI + InsightFace                          │
│                                                                   │
│   buffalo_l:    SCRFD-10G → align → ResNet50  ArcFace → 512-dim   │
│   antelopev2:   SCRFD-10G → align → ResNet100 ArcFace → 512-dim   │
│                                                                   │
│   Both models run concurrently (asyncio.gather).                  │
└──────────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│   UI shows three cards side-by-side:                              │
│   [Frontend MobileFaceNet] [Backend buffalo_l] [Backend antelopev2]│
└──────────────────────────────────────────────────────────────────┘
```

The backend is optional; if it's offline, the UI just marks those cards "Backend offline" and the frontend pipeline keeps running.

### Model lineup

| Tier     | Model                     | Architecture      | Size   | Runs on          |
| -------- | ------------------------- | ----------------- | ------ | ---------------- |
| Frontend | MobileFaceNet (w600k_mbf) | MobileFaceNet     | 13 MB  | User's browser   |
| Backend  | buffalo_l                 | ResNet-50 ArcFace | 182 MB | Your server      |
| Backend  | antelopev2                | ResNet-100 ArcFace| 264 MB | Your server      |

All three use the same training method (ArcFace loss) and share the WebFace600K training-data lineage. Only the architecture/size differs.

---

## Project Structure

### Frontend (`src/`)

```
src/
  app/
    page.tsx                        Main page (SSR disabled, dynamic import)
    layout.tsx                      Root layout
    globals.css                     Tailwind CSS
  components/
    FaceComparison.tsx              Orchestrator (composes hooks + sub-components)
    BackendBadge.tsx                Status pill (checking / online / offline)
    ModelCard.tsx                   Per-model similarity/status card
  hooks/
    useFaceModels.ts                Loads ONNX models once
    useBackendHealth.ts             Polls /api/health; exposes status
    useWebcamStream.ts              Acquires and tears down MediaStream
    useDetectionOverlay.ts          Draws box + landmarks on canvas
    useProfileUpload.ts             Reads/resizes uploaded image, extracts embedding
    useLiveDetection.ts             Periodic detect → embed → compare → call backend
  lib/
    config.ts                       Shared constants (threshold, intervals, BACKEND_URL)
    faceApi.ts                      ONNX inference: detect / align / embed
    backend.ts                      Typed HTTP client for the Python backend

public/
  models/
    det_500m.onnx                   SCRFD face detector (2.4 MB)
    w600k_mbf.onnx                  MobileFaceNet ArcFace embedder (13 MB)
```

### Backend (`python/`)

```
python/
  main.py                           Entrypoint (app factory, lifespan, CORS)
  config.py                         Constants + logging setup
  schemas.py                        DTOs (VerifyRequest/Response, ModelResult, ...)
  services/
    model_registry.py               Eager + lock-guarded lazy model loading
    face_service.py                 compare_faces() — runs inference off-thread
  routers/
    verify.py                       POST /api/verify
    health.py                       GET  /api/health  +  GET /
  utils/
    image.py                        decode_base64_image()
  requirements.txt
```

---

## How It Works

### The four steps

```
1. DETECTION    Find WHERE a face is (bounding box + 5 landmarks)
                Model: SCRFD-500M (frontend) / SCRFD-10G (backend)

2. ALIGNMENT    Rotate/scale the face so eyes, nose, mouth are at canonical
                112×112 positions — required for ArcFace to work reliably.

3. EMBEDDING    Convert the aligned face into a 512-number vector (fingerprint)
                Model: MobileFaceNet (frontend) / ResNet50/100 ArcFace (backend)

4. COMPARISON   Cosine similarity between two embeddings
                Above the 0.4 threshold = Match
```

### Every 2.5 seconds the frontend

1. Snapshots the webcam frame (mirrored to match selfie orientation).
2. Runs detect → align → embed locally with MobileFaceNet — instant.
3. In parallel, POSTs both images to `/api/verify`.
4. The backend runs buffalo_l and antelopev2 concurrently.
5. UI updates all three cards.

The backend call is non-blocking and rate-limited to one in-flight request at a time — if a request takes longer than 2.5 s the next frontend cycle still runs without waiting.

### Expected similarity ranges

| Scenario                                                   | Same-condition pairs | Photo vs webcam |
| ---------------------------------------------------------- | -------------------: | --------------: |
| Same photo vs itself                                       | 95–100 %             | —               |
| Same person, same camera/lighting                          | 70–90 %              | —               |
| Same person, different camera/lighting (photo vs webcam)   | —                    | 40–60 %         |
| Different person                                           | 5–25 %               | 5–25 %          |

Typical differences by tier when comparing a photo to a live webcam:

| Model                        | Same person | Different person |
| ---------------------------- | ----------: | ---------------: |
| Frontend MobileFaceNet       | 45–55 %     | 5–25 %           |
| Backend buffalo_l (ResNet50) | 55–65 %     | 3–20 %           |
| Backend antelopev2 (ResNet100)| 60–70 %    | 2–15 %           |

The important metric isn't the absolute similarity — it's the **gap** between same-person and different-person scores. Bigger models give wider gaps and therefore more reliable matching.

---

## API Reference

### `POST /api/verify`

Compare two face images using both backend models.

**Request**

```json
{
  "profile_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "live_image":    "data:image/jpeg;base64,/9j/4AAQ..."
}
```

Either raw base64 or full data URI works.

**Response**

```json
{
  "buffalo_l": {
    "similarity": 0.65,
    "verified": true,
    "threshold": 0.4,
    "took_ms": 250,
    "error": null
  },
  "antelopev2": {
    "similarity": 0.72,
    "verified": true,
    "threshold": 0.4,
    "took_ms": 450,
    "error": null
  }
}
```

If a face can't be detected in one of the images, that model's result has `similarity: null` and `error: "no_face_in_profile"` or `"no_face_in_live"`.

### `GET /api/health`

```json
{
  "status": "ok",
  "models": [
    { "name": "buffalo_l",  "description": "SCRFD-10G + ResNet50 ArcFace",  "size_mb": 182 },
    { "name": "antelopev2", "description": "SCRFD-10G + ResNet100 ArcFace", "size_mb": 264 }
  ],
  "threshold": 0.4
}
```

### `GET /docs`

Auto-generated Swagger UI.

---

## Configuration

### Backend URL

Frontend calls `http://localhost:8000` by default. Override with:

```bash
# .env.local at project root
NEXT_PUBLIC_BACKEND_URL=http://192.168.1.100:8000
```

### Backend port

```bash
uvicorn main:app --reload --port 8001
```

Don't forget to update `NEXT_PUBLIC_BACKEND_URL` to match.

### Match threshold

Both tiers use cosine similarity ≥ 0.4. Change it in:

- Frontend: `src/lib/config.ts` → `MATCH_THRESHOLD`
- Backend: `python/config.py` → `MATCH_THRESHOLD`

---

## Performance

### Frontend (browser WASM)

Approximate per-frame times for detect + embed:

| User's device                  | Detection | Embedding | Total     |
| ------------------------------ | --------: | --------: | --------: |
| Modern laptop (M1/M2, i7)      | ~200 ms   | ~100 ms   | ~300 ms   |
| Older laptop (2018 era)        | ~500 ms   | ~300 ms   | ~800 ms   |
| Mid-range phone                | ~400 ms   | ~250 ms   | ~650 ms   |
| Low-end phone                  | ~1000 ms  | ~600 ms   | ~1600 ms  |

### Backend (CPU)

Per-request time (both models run concurrently; total ≈ slower of the two):

| CPU cores     | buffalo_l | antelopev2 | Total (concurrent) |
| ------------- | --------: | ---------: | -----------------: |
| 1 vCPU        | ~800 ms   | ~1500 ms   | ~1500 ms           |
| 2 vCPU        | ~500 ms   | ~900 ms    | ~900 ms            |
| 4 vCPU        | ~300 ms   | ~550 ms    | ~550 ms            |
| 8 vCPU        | ~200 ms   | ~350 ms    | ~350 ms            |
| 1× NVIDIA T4  | ~30 ms    | ~50 ms     | ~50 ms             |
| 1× NVIDIA A100| ~10 ms    | ~15 ms     | ~15 ms             |

If a request takes longer than 2.5 s, the next frontend cycle skips the backend call (one in-flight request at a time).

---

## Troubleshooting

**Backend cards stay "Backend offline" forever.** Check the server: `curl http://localhost:8000/api/health`. Then open DevTools → Network and look for failed requests to `/api/verify`. CORS is permissive by default.

**First backend request takes minutes.** InsightFace downloads models to `~/.insightface/models/` on first use (~450 MB). Subsequent requests hit the local cache.

**Models fail to download.** Check your internet connection. You can pre-download manually:

```python
import insightface
insightface.app.FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1, det_size=(640, 640))
insightface.app.FaceAnalysis(name='antelopev2').prepare(ctx_id=-1, det_size=(640, 640))
```

**`Address already in use`.** Port 8000 is taken. Pick a different one (`uvicorn main:app --port 8001`) and update `NEXT_PUBLIC_BACKEND_URL`.

**Out of memory.** Both models together use ~2 GB RAM. On a small VPS, remove `antelopev2` from `EAGER_MODELS` in `config.py` or comment out its call in `routers/verify.py`.

**`ModuleNotFoundError: No module named 'insightface'`.** Activate the virtualenv (`source venv/bin/activate`) before running `uvicorn`.

---

## Frontend vs Backend

### Frontend-only

**Pros:** zero server cost, no network latency, privacy (data never leaves the browser), static deployment (Vercel/Netlify), works offline after first download.

**Cons:** limited to smaller models (13 MB, not 248 MB), ~15 MB first-visit download, speed depends on the user's device, no anti-spoofing/liveness, 1:1 comparison only (not 1:N database search), client code is inspectable.

### Backend

**Pros:** runs larger, more accurate models; ~10–20 % higher similarity for same person; better edge cases (angles, lighting, occlusion); room for liveness/anti-spoofing and 1:N search.

**Cons:** server cost scales with users, network latency every request, face data leaves the device.

### Hybrid (recommended for production)

```
Frontend (onnxruntime-web)
  • Face detection (lightweight)
  • Face alignment
  • Embedding extraction (MobileFaceNet, runs on user's device)

Backend (FastAPI)
  • Receives embeddings only (~2 KB), not images
  • Vector comparison / 1:N search (microseconds)
  • Liveness / anti-spoofing
  • Stores embeddings in a vector DB (pgvector, Qdrant)
```

Heavy inference runs on the user's device (free). The backend only does lightweight comparison and storage. Scales to thousands of users on a small server.

### Suggested production stack

| Component          | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| API framework      | FastAPI (Python)                                        |
| Face recognition   | DeepFace or ONNX Runtime + ArcFace                      |
| Vector database    | PostgreSQL + pgvector, or Qdrant                        |
| Liveness detection | Azure Face API or Silent-Face-Anti-Spoofing             |

---

# Knowledge Base

Everything below is background reference material — safe to skip if you just want to run the app.

## Models used in this project

### `det_500m.onnx` — face detection

| Field            | Value                                                     |
| ---------------- | --------------------------------------------------------- |
| Full name        | SCRFD-500M                                                |
| What it does     | Detects faces; outputs bounding box + 5 landmarks         |
| "500M" means     | 500 thousand parameters (M = Mille)                       |
| Size             | 2.4 MB                                                    |
| Input            | 640×640 image                                             |
| Output           | Boxes, scores, 5 landmarks per face                       |
| Created by       | InsightFace                                               |
| Architecture     | SCRFD (Sample and Computation Redistribution)             |

**Other SCRFD variants**

| Variant         | Parameters | Size   | Use case        |
| --------------- | ---------: | -----: | --------------- |
| SCRFD-500M (ours)| 500K      | 2.4 MB | Browser/mobile  |
| SCRFD-2.5G      | 2.5M       | ~10 MB | Edge devices    |
| SCRFD-10G       | 10M        | ~16 MB | Server          |

### `w600k_mbf.onnx` — face recognition

| Field           | Value                                                                 |
| --------------- | --------------------------------------------------------------------- |
| Full name       | WebFace600K MobileFaceNet                                             |
| What it does    | Converts an aligned face into a 512-dim embedding                     |
| "w600k" means   | Trained on WebFace600K (600K identities, ~10M images)                 |
| "mbf" means     | MobileFaceNet architecture                                            |
| Training method | ArcFace loss                                                          |
| Size            | 13 MB                                                                 |
| Input           | 112×112 RGB aligned face                                              |
| Output          | 512 floats                                                            |
| Accuracy        | 99.70 % LFW, 98.00 % CFP-FP, 96.58 % AgeDB-30                         |
| Created by      | InsightFace                                                           |

Decoded: **w600k_mbf** = WebFace600K-trained MobileFaceNet with ArcFace loss.

**Other InsightFace recognition variants**

| File                | Decoded name               | Architecture | Training data | Size    | LFW    |
| ------------------- | -------------------------- | ------------ | ------------- | ------: | -----: |
| w600k_mbf (ours)    | WebFace600K MobileFaceNet  | MobileFaceNet| WebFace600K   | 13 MB   | 99.70% |
| w600k_r50           | WebFace600K ResNet50       | ResNet-50    | WebFace600K   | ~166 MB | 99.80% |
| glint360k_r100      | Glint360K ResNet100        | ResNet-100   | Glint360K     | ~248 MB | 99.82% |

### Naming convention

InsightFace files follow `{dataset}_{architecture}.onnx`:

```
w600k_mbf.onnx         → WebFace600K + MobileFaceNet
w600k_r50.onnx         → WebFace600K + ResNet-50
glint360k_r100.onnx    → Glint360K + ResNet-100
det_500m.onnx          → detection model, 500K parameters
```

## Landmarks

A landmark is a specific (x, y) coordinate point on a detected face.

**5-point landmarks** (used here)

```
     (0) Left Eye            (1) Right Eye

              (2) Nose Tip

     (3) Left Mouth          (4) Right Mouth
```

**68-point landmarks** — jawline, eyebrows, nose, eyes, mouth — used for expression detection, face morphing, etc. Not needed for recognition.

**Why they matter:** without landmarks you can only crop a rough box. With them you can **align** the face precisely, which is critical for recognition. Going from no alignment to landmark-based alignment moved same-person similarity from ~20 % to ~50 % in early development of this project.

## Face alignment

Alignment transforms a face so the eyes/nose/mouth land at fixed pixel positions in a 112×112 image.

ArcFace models were trained on millions of aligned faces where:

```
Left eye    (38.29, 51.70)
Right eye   (73.53, 51.50)
Nose        (56.03, 71.74)
Left mouth  (41.55, 92.37)
Right mouth (70.73, 92.20)
```

Without alignment, features land in unexpected positions and the model produces unreliable embeddings.

**How it works:**

1. Detect 5 landmarks.
2. Compute a similarity transform (rotation + uniform scale + translation) that maps the detected landmarks to the canonical positions.
3. Apply the transform and crop to 112×112.

## Embeddings

An embedding is a fixed-length vector that represents a face. Think numerical fingerprint.

```
Face image → Model → [0.12, -0.45, 0.78, -0.23, …, 0.56]   (512 numbers)
```

- Same person → nearby embeddings.
- Different person → distant embeddings.
- Individual numbers aren't human-interpretable — the model learned which combinations matter.

**Comparing two embeddings**

| Method             | Formula                          | Same person       | Different person |
| ------------------ | -------------------------------- | ----------------- | ---------------- |
| Cosine similarity  | dot(a, b) / (‖a‖·‖b‖)            | Close to 1        | Close to 0       |
| Euclidean distance | √(Σ (a − b)²)                    | Close to 0        | Large            |

This project uses cosine similarity. Both vectors are L2-normalized, so cosine similarity equals the dot product.

## Recognition models compared

| Model     | Year | Training method    | Best LFW | Speed  | Verdict                |
| --------- | ---: | ------------------ | -------: | ------ | ---------------------- |
| ArcFace   | 2018 | Angular margin     | 99.82 %  | Medium | Best overall           |
| FaceNet512| 2015 | Triplet loss       | 99.65 %  | Medium | Solid alternative      |
| CosFace   | 2018 | Cosine margin      | 99.73 %  | Medium | Very close to ArcFace  |
| SFace     | 2021 | Sigmoid loss       | 99.60 %  | Fast   | Lightweight option     |
| VGG-Face  | 2015 | Softmax            | 98.78 %  | Slow   | Outdated               |
| face-api.js (dlib) | 2017 | —         | ~95 %    | Fast   | Demos only             |

**ArcFace** is a training method (loss function), not a specific architecture. Any backbone — MobileFaceNet, ResNet50, ResNet100 — can be trained with ArcFace loss.

## Detection models

| Model       | Output                    | Size    | Notes                                                          |
| ----------- | ------------------------- | ------: | -------------------------------------------------------------- |
| SCRFD       | Box + 5 landmarks         | 2–16 MB | Efficient; multiple size variants (used in this project)        |
| RetinaFace  | Box + 5 landmarks         | ~100 MB | Very accurate; too big for browsers                             |
| MTCNN       | Box + 5 landmarks         | small   | 3-stage cascade (P/R/O-Net)                                     |
| UltraFace   | Box only                  | ~1 MB   | No landmarks → can't align → bad recognition                    |
| SSD         | Box only                  | varies  | Generic object detector adapted for faces                       |

## Benchmarks & datasets

### Benchmarks (test sets)

| Benchmark  | What it tests                                                 |
| ---------- | ------------------------------------------------------------- |
| **LFW**    | 6 000 pairs of "in the wild" faces; de-facto sanity check.    |
| **CFP-FP** | Frontal vs profile matching — much harder than LFW.           |
| **AgeDB-30** | Same person with a ≥30-year age gap.                        |

LFW is largely "solved" — most modern models hit 99.5 %+, so it no longer discriminates between top models.

### Training datasets

| Dataset       | Identities | Images  | Used by                   |
| ------------- | ---------: | ------: | ------------------------- |
| WebFace600K   | 600 000    | ~10 M   | Our frontend model (w600k_mbf) |
| MS1MV2        | 85 000     | 5.8 M   | Many ArcFace variants     |
| Glint360K     | 360 000    | 17 M    | Largest ResNet100 models  |
| VGGFace2      | 9 131      | 3.3 M   | VGG-Face                  |
| CASIA-WebFace | 10 575     | 494 414 | Older models              |

More identities → better generalization. WebFace600K outperforms CASIA-WebFace on nearly everything.

## Libraries & formats

### Libraries

- **DeepFace (Python)** — wrapper around existing models (ArcFace, FaceNet512, VGG-Face, SFace, …). Handy one-liner API.
- **InsightFace (Python)** — the toolkit from the team that created ArcFace and SCRFD. Our backend uses it.
- **face-api.js** — older browser library using dlib ResNet (~2017). ~95 % LFW; no longer maintained.
- **ONNX Runtime Web** — general-purpose ONNX runner for the browser (via WebAssembly). This project uses it to run SCRFD + MobileFaceNet on the client.

### ONNX

| Term             | Meaning                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| ONNX             | Universal AI model file format (like `.pdf` for documents)             |
| ONNX Runtime     | The engine that runs `.onnx` files                                     |
| onnxruntime-web  | Browser flavor (WebAssembly)                                           |

**Why ONNX:** one model file runs anywhere — browser, server, mobile — independent of PyTorch/TensorFlow at inference time.

Other model formats: `.pt`/`.pth` (PyTorch), `.h5`/`.keras` (TF/Keras), `.pb` (TF SavedModel), `.tflite` (mobile). All convert to ONNX.

## Training concepts

### Loss functions

| Loss                  | Used by    | Idea                                                                              |
| --------------------- | ---------- | --------------------------------------------------------------------------------- |
| ArcFace (angular)     | ArcFace    | Adds an angular margin in feature space — currently the best.                     |
| Triplet loss          | FaceNet    | Compares anchor/positive/negative; pulls positives closer, pushes negatives away. |
| CosFace (cosine)      | CosFace    | Like ArcFace but the margin is in cosine rather than angular space.                |
| Softmax               | VGG-Face   | Pure classification; less discriminative for embeddings.                          |

### Architectures

The "body" of the network. Same architecture can be trained with different losses and datasets.

| Architecture     | Params | Size    | Speed    | Use case                  |
| ---------------- | -----: | ------: | -------- | ------------------------- |
| MobileFaceNet    | ~1 M   | 13 MB   | Fast     | Mobile / browser          |
| ResNet-50        | ~25 M  | ~166 MB | Medium   | Server                    |
| ResNet-100       | ~65 M  | ~248 MB | Slow     | Server (max accuracy)     |
| Inception-ResNet | ~23 M  | ~100 MB | Medium   | FaceNet                   |
| VGG-16           | ~138 M | ~500 MB | Very slow| Legacy (VGG-Face)         |

### How a face-recognition model is built

```
1. Choose ARCHITECTURE:   MobileFaceNet (small, fast)
2. Choose DATASET:        WebFace600K (600K identities)
3. Choose LOSS:           ArcFace
4. Train for days/weeks on GPUs
5. Export to ONNX:        w600k_mbf.onnx
```

## Who pays for compute?

Where a model runs determines who foots the bill.

**Frontend** — model ships with the website, runs on the user's device. Your server just serves static files. Cost: ~$0.

**Backend** — server runs the model for every request. Cost: $50–$500+/month depending on hardware and traffic.

**Hybrid** — the user's device does the expensive inference; the server just does vector math/search. Cost: ~$5–$30/month.

### Cost sketch by scale

| Scenario         | Frontend-only | Hybrid      | Backend (CPU, ResNet100) | Backend (GPU) |
| ---------------- | ------------: | ----------: | -----------------------: | ------------: |
| 100 users/day    | $0            | $5–15       | $30–60                   | $250–400      |
| 5 000 users/day  | $0–20         | $5–15       | $120–250                 | $250–400      |
| 100 000 users/day| $20–50        | $10–30      | $500–1 000               | $800–1 400    |

### When to use what

| Your situation                   | Recommended           | Why                                            |
| -------------------------------- | --------------------- | ---------------------------------------------- |
| MVP / prototype                  | Frontend only         | $0 cost, fast to build                         |
| < 1 000 users/day                | Frontend only         | Still $0                                       |
| Need bigger model accuracy       | Backend with GPU      | Only way to run ResNet100                      |
| Need 1:N database search         | Hybrid                | Best of both worlds                            |
| Privacy critical                 | Frontend only         | Face data never leaves the device              |
| Users on slow devices            | Backend               | Server does the heavy work                     |
| 10 000+ concurrent               | Backend (multi-GPU)   | Predictable performance                        |

### Key insight: the math is free

```
Face detection:  ~5–500 ms    ← expensive
Face embedding:  ~2–800 ms    ← expensive
Face comparison: ~0.01 ms     ← basically free
```

The AI model is the expensive part. Comparing two 512-number vectors is just multiplies and adds — any server can do billions per second. That's why the hybrid approach scales so well.

## Known considerations

- **Mirroring.** Raw webcam frames are not mirrored, but selfies typically are. The app flips the webcam frame to match selfie orientation.
- **Cross-condition similarity.** Photo vs webcam will always be lower (40–60 %) than same-condition comparisons (70–90 %). This is expected, not a bug.
- **Model download.** First visit downloads ~15 MB of model files. Subsequent visits use the browser cache.
- **No alignment = bad accuracy.** Without landmark-based alignment, even the best recognition model produces unreliable embeddings.

---

## Glossary

| Term | Definition |
| ---- | ---------- |
| **AgeDB**               | Benchmark testing face recognition across age gaps. |
| **Alignment**           | Transforming a face so features land at canonical pixel positions. |
| **ArcFace**             | State-of-the-art face recognition training method (angular margin loss). |
| **Backbone**            | The main neural network architecture (ResNet50, MobileFaceNet, …). |
| **Bounding box**        | Rectangle around a detected face (x1, y1, x2, y2). |
| **CFP-FP**              | Benchmark testing frontal vs profile face recognition. |
| **Cosine similarity**   | Similarity between two vectors (1 = identical, 0 = unrelated). |
| **DeepFace**            | Python wrapper library for face recognition (not a model). |
| **Detection**           | Finding where faces are in an image. |
| **Embedding**           | Fixed-length vector representing a face (face fingerprint). |
| **Euclidean distance**  | Alternative similarity measure (lower = more similar). |
| **face-api.js**         | Older JavaScript face recognition library. |
| **FaceNet**             | Google's face recognition model using triplet loss. |
| **Glint360K**           | Large training dataset (360K identities). |
| **InsightFace**         | Team/toolkit that created ArcFace and SCRFD. |
| **Landmark**            | Specific coordinate point on a face (eye, nose, mouth). |
| **LFW**                 | Labeled Faces in the Wild — the standard benchmark. |
| **Loss function**       | Mathematical function that guides how a model learns. |
| **MobileFaceNet**       | Lightweight face recognition architecture for mobile/edge. |
| **MTCNN**               | 3-stage face detection model. |
| **NMS**                 | Non-Maximum Suppression — removes duplicate detections. |
| **ONNX**                | Open Neural Network Exchange — universal model file format. |
| **ONNX Runtime**        | Engine that runs ONNX files. |
| **Recognition**         | Identifying WHO a face belongs to (by comparing embeddings). |
| **ResNet**              | Deep neural network architecture with skip connections. |
| **RetinaFace**          | High-accuracy face detector with landmarks. |
| **SCRFD**               | Efficient face detector with landmarks (used in this project). |
| **Similarity transform**| Rotation + scale + translation, used for alignment. |
| **Softmax**             | Classification training approach. |
| **Triplet loss**        | Training method comparing anchor, positive, and negative samples. |
| **UltraFace**           | Ultra-lightweight face detector (no landmarks). |
| **VGG-Face**            | Oxford's face recognition model using VGG-16 architecture. |
| **WASM**                | WebAssembly — near-native code speed in browsers. |
| **WebFace600K**         | Training dataset with 600K face identities (~10M images). |
