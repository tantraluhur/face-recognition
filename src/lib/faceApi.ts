import * as ort from "onnxruntime-web";

// ── ONNX Runtime configuration ─────────────────────────────────────────
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

// ── Model constants ────────────────────────────────────────────────────
const DET_SIZE = 640;
const REC_SIZE = 112;
const SCORE_THRESH = 0.5;
const NMS_THRESH = 0.4;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;
const NUM_LANDMARKS = 5;

// Canonical landmark positions for ArcFace 112×112 alignment.
const ARCFACE_DST: ReadonlyArray<readonly [number, number]> = [
  [38.2946, 51.6963], // left eye
  [73.5318, 51.5014], // right eye
  [56.0252, 71.7366], // nose
  [41.5493, 92.3655], // left mouth
  [70.7299, 92.2041], // right mouth
];

// ── Public types ───────────────────────────────────────────────────────
export interface Detection {
  /** Bounding box in original image coordinates: [x1, y1, x2, y2]. */
  box: [number, number, number, number];
  score: number;
  /** Five landmarks (LE, RE, nose, LM, RM) in original image coordinates. */
  landmarks: number[][];
}

export interface EmbeddingResult {
  embedding: Float32Array;
  /** Base64 PNG of the aligned 112×112 face (useful for UI debug display). */
  alignedFaceDataUrl: string;
}

type ImageSource = HTMLVideoElement | HTMLCanvasElement;

// ── Model loading ──────────────────────────────────────────────────────
let detSession: ort.InferenceSession | null = null;
let recSession: ort.InferenceSession | null = null;

export async function loadModels(): Promise<void> {
  [detSession, recSession] = await Promise.all([
    ort.InferenceSession.create("/models/det_500m.onnx"),
    ort.InferenceSession.create("/models/w600k_mbf.onnx"),
  ]);
}

// ── Face detection (SCRFD-500M) ────────────────────────────────────────
// Input:  input.1 [1, 3, 640, 640] — (pixel − 127.5) / 128
// Output: 9 tensors — scores/bboxes/keypoints at strides 8, 16, 32.
export async function detectFace(source: ImageSource): Promise<Detection | null> {
  if (!detSession) throw new Error("Models not loaded");

  const { canvas, scale, padX, padY } = letterbox(source, DET_SIZE);
  const imageData = getImageData(canvas);
  const input = preprocess(imageData, 127.5, 128);

  const results = await detSession.run({ "input.1": input });
  const detections = decodeSCRFD(results, detSession.outputNames, scale, padX, padY);
  if (detections.length === 0) return null;

  // Pick the largest face (closest to camera) rather than the highest-scoring,
  // so a confident background face doesn't beat the person in front.
  return detections.reduce((best, d) => (area(d.box) > area(best.box) ? d : best));
}

function letterbox(source: ImageSource, size: number) {
  const srcW = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = (size - newW) / 2;
  const padY = (size - newH) / 2;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = requireCtx(canvas);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, newW, newH);

  return { canvas, scale, padX, padY };
}

function decodeSCRFD(
  results: ort.InferenceSession.ReturnType,
  names: readonly string[],
  scale: number,
  padX: number,
  padY: number,
): Detection[] {
  const fmc = STRIDES.length;
  const all: Detection[] = [];

  for (let idx = 0; idx < fmc; idx++) {
    const stride = STRIDES[idx];
    const scores = results[names[idx]].data as Float32Array;
    const bboxes = results[names[idx + fmc]].data as Float32Array;
    const kpss = results[names[idx + fmc * 2]].data as Float32Array;
    const feat = DET_SIZE / stride;

    for (let y = 0; y < feat; y++) {
      for (let x = 0; x < feat; x++) {
        for (let a = 0; a < NUM_ANCHORS; a++) {
          const ai = (y * feat + x) * NUM_ANCHORS + a;
          if (scores[ai] < SCORE_THRESH) continue;

          const cx = x * stride;
          const cy = y * stride;

          const bi = ai * 4;
          const box: Detection["box"] = [
            (cx - bboxes[bi] * stride - padX) / scale,
            (cy - bboxes[bi + 1] * stride - padY) / scale,
            (cx + bboxes[bi + 2] * stride - padX) / scale,
            (cy + bboxes[bi + 3] * stride - padY) / scale,
          ];

          const ki = ai * NUM_LANDMARKS * 2;
          const landmarks: number[][] = [];
          for (let k = 0; k < NUM_LANDMARKS; k++) {
            landmarks.push([
              (cx + kpss[ki + k * 2] * stride - padX) / scale,
              (cy + kpss[ki + k * 2 + 1] * stride - padY) / scale,
            ]);
          }

          all.push({ box, score: scores[ai], landmarks });
        }
      }
    }
  }

  return nms(all, NMS_THRESH);
}

function nms(dets: Detection[], thresh: number): Detection[] {
  dets.sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  const dead = new Set<number>();
  for (let i = 0; i < dets.length; i++) {
    if (dead.has(i)) continue;
    kept.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (!dead.has(j) && iou(dets[i].box, dets[j].box) > thresh) dead.add(j);
    }
  }
  return kept;
}

function iou(a: number[], b: number[]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  return inter / (area(a as Detection["box"]) + area(b as Detection["box"]) - inter);
}

function area(box: Detection["box"]): number {
  return (box[2] - box[0]) * (box[3] - box[1]);
}

// ── Face alignment + embedding ─────────────────────────────────────────
// Uses 5 landmarks to compute a similarity transform → aligned 112×112 face.
// Then runs MobileFaceNet (ArcFace) to extract a 512-dim embedding.
export async function extractEmbedding(
  source: ImageSource,
  det: Detection,
): Promise<EmbeddingResult> {
  if (!recSession) throw new Error("Models not loaded");

  const aligned = alignFace(source, det.landmarks);
  const alignedFaceDataUrl = aligned.toDataURL("image/png");

  const imageData = getImageData(aligned);
  const input = preprocess(imageData, 127.5, 127.5);

  const results = await recSession.run({ "input.1": input });
  const embedding = results["516"].data as Float32Array;
  return { embedding: l2Normalize(embedding), alignedFaceDataUrl };
}

function alignFace(source: ImageSource, landmarks: number[][]): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = REC_SIZE;
  canvas.height = REC_SIZE;
  const ctx = requireCtx(canvas);

  // Similarity transform mapping source landmarks → canonical ArcFace positions.
  //   canvasX = a·srcX − b·srcY + tx
  //   canvasY = b·srcX + a·srcY + ty
  const [a, b, tx, ty] = estimateSimilarityTransform(landmarks, ARCFACE_DST);
  ctx.setTransform(a, b, -b, a, tx, ty);
  ctx.drawImage(source, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  return canvas;
}

/**
 * Estimate a 2D similarity transform (rotation + uniform scale + translation)
 * via least-squares Procrustes. Returns [a, b, tx, ty] such that:
 *   dstX = a·srcX − b·srcY + tx
 *   dstY = b·srcX + a·srcY + ty
 */
function estimateSimilarityTransform(
  src: number[][],
  dst: ReadonlyArray<readonly [number, number]>,
): [number, number, number, number] {
  const n = src.length;

  let smx = 0, smy = 0, dmx = 0, dmy = 0;
  for (let i = 0; i < n; i++) {
    smx += src[i][0];
    smy += src[i][1];
    dmx += dst[i][0];
    dmy += dst[i][1];
  }
  smx /= n; smy /= n; dmx /= n; dmy /= n;

  let varS = 0, covA = 0, covB = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - smx;
    const sy = src[i][1] - smy;
    const dx = dst[i][0] - dmx;
    const dy = dst[i][1] - dmy;
    varS += sx * sx + sy * sy;
    covA += sx * dx + sy * dy;
    covB += sx * dy - sy * dx;
  }

  const a = covA / varS;
  const b = covB / varS;
  const tx = dmx - (a * smx - b * smy);
  const ty = dmy - (b * smx + a * smy);
  return [a, b, tx, ty];
}

// ── Preprocessing & utilities ──────────────────────────────────────────

/**
 * Convert RGBA ImageData to a [1, 3, H, W] float tensor with per-pixel
 * normalization `(pixel - mean) / std`. Used for both SCRFD (mean=127.5,
 * std=128) and ArcFace (mean=127.5, std=127.5).
 */
function preprocess(imageData: ImageData, mean: number, std: number): ort.Tensor {
  const { data, width, height } = imageData;
  const n = width * height;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    f[i] = (data[j] - mean) / std;
    f[n + i] = (data[j + 1] - mean) / std;
    f[2 * n + i] = (data[j + 2] - mean) / std;
  }
  return new ort.Tensor("float32", f, [1, 3, height, width]);
}

function getImageData(canvas: HTMLCanvasElement): ImageData {
  return requireCtx(canvas).getImageData(0, 0, canvas.width, canvas.height);
}

function requireCtx(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas context unavailable");
  return ctx;
}

function l2Normalize(v: Float32Array): Float32Array {
  let sq = 0;
  for (let i = 0; i < v.length; i++) sq += v[i] * v[i];
  const norm = Math.sqrt(sq);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/** Cosine similarity. Inputs must be L2-normalized. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}
