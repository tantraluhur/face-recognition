// Pure helpers that turn MediaPipe Face Landmarker outputs into the signals
// our UI cares about: head-pose angles and a single gaze vector.
//
// All heavy lifting — landmark detection, blendshape scoring, the pose
// transformation matrix — is done by MediaPipe. This file only:
//   1. Reads yaw/pitch/roll out of the 4×4 matrix MediaPipe hands us.
//   2. Combines MediaPipe's eye-look blendshape scores into one 2-D vector.
//   3. Thresholds those numbers into an "on-screen / looking away / …" status.

import type { Category, Classifications, Matrix } from "@mediapipe/tasks-vision";

export interface HeadPose {
  /** Degrees. +right / –left (head shaking "no"). */
  yaw: number;
  /** Degrees. +up / –down (head nodding "yes"). */
  pitch: number;
  /** Degrees. +tilt-right / –tilt-left (head tilted shoulder-to-shoulder). */
  roll: number;
}

export interface Gaze {
  /** Normalized. +right / –left (subject's perspective). Range ≈ [-1, 1]. */
  x: number;
  /** Normalized. +up / –down. Range ≈ [-1, 1]. */
  y: number;
}

export type AttentionStatus =
  | "on-screen"
  | "looking-away"
  | "eyes-closed"
  | "no-face";

export interface AttentionState {
  status: AttentionStatus;
  headPose: HeadPose | null;
  gaze: Gaze | null;
  eyesClosed: boolean;
}

/**
 * A user's personal "zero" — their resting gaze and head pose when they are
 * actually looking at the camera. Subtracted from every subsequent reading
 * so the thresholds in `decideStatus` measure *deviation* from normal, not
 * the raw MediaPipe vector (which is never `(0,0)` for real users).
 */
export interface AttentionBaseline {
  gaze: Gaze;
  yaw: number;
  pitch: number;
}

export interface AttentionSample {
  gaze: Gaze;
  headPose: HeadPose;
}

export const NO_FACE: AttentionState = {
  status: "no-face",
  headPose: null,
  gaze: null,
  eyesClosed: false,
};

// ── Thresholds ─────────────────────────────────────────────────────────
// Tune these for your lighting / camera. These defaults are a sensible start
// for a "user is roughly paying attention" check.
const HEAD_AWAY_DEG = 20;   // |yaw| or |pitch| above this counts as looking away
const GAZE_AWAY = 0.35;     // |gaze| vector magnitude above this counts as away
const EYES_CLOSED = 0.55;   // averaged eye-blink blendshape score

// ── MediaPipe matrix layout ────────────────────────────────────────────
// `Matrix.data` is a flat array of 16 floats in column-major order:
//   data[0]=m00  data[4]=m01  data[8] =m02  data[12]=m03
//   data[1]=m10  data[5]=m11  data[9] =m12  data[13]=m13
//   data[2]=m20  data[6]=m21  data[10]=m22  data[14]=m23
//   data[3]=m30  data[7]=m31  data[11]=m32  data[15]=m33
//
// We only need the 3×3 rotation block. Standard ZYX-order Euler extraction.
export function headPoseFromMatrix(matrix: Matrix): HeadPose {
  const d = matrix.data;
  const m02 = d[8];
  const m12 = d[9];
  const m22 = d[10];
  const m10 = d[1];
  const m11 = d[5];

  const toDeg = 180 / Math.PI;
  return {
    yaw: Math.atan2(m02, m22) * toDeg,
    pitch: Math.asin(-clamp(m12, -1, 1)) * toDeg,
    roll: Math.atan2(m10, m11) * toDeg,
  };
}

// ── Gaze from blendshape scores ────────────────────────────────────────
// "InLeft"  means the LEFT eye looking toward the nose (to subject's right).
// "OutLeft" means the LEFT eye looking away from the nose (subject's left).
// Subject's right = +x, subject's up = +y in our Gaze convention.
export function gazeFromBlendshapes(blendshapes: Classifications): Gaze {
  const s = toScoreMap(blendshapes.categories);
  const x =
    (s.eyeLookInLeft ?? 0) + (s.eyeLookOutRight ?? 0) -
    (s.eyeLookOutLeft ?? 0) - (s.eyeLookInRight ?? 0);
  const y =
    (s.eyeLookUpLeft ?? 0) + (s.eyeLookUpRight ?? 0) -
    (s.eyeLookDownLeft ?? 0) - (s.eyeLookDownRight ?? 0);
  return { x: clamp(x / 2, -1, 1), y: clamp(y / 2, -1, 1) };
}

export function eyesClosedFromBlendshapes(blendshapes: Classifications): boolean {
  const s = toScoreMap(blendshapes.categories);
  const avg = ((s.eyeBlinkLeft ?? 0) + (s.eyeBlinkRight ?? 0)) / 2;
  return avg >= EYES_CLOSED;
}

// ── Top-level combine ──────────────────────────────────────────────────
export function attentionFromResult(params: {
  matrix: Matrix | undefined;
  blendshapes: Classifications | undefined;
  baseline?: AttentionBaseline | null;
}): AttentionState {
  const { matrix, blendshapes, baseline } = params;
  if (!matrix || !blendshapes) return NO_FACE;

  const rawHeadPose = headPoseFromMatrix(matrix);
  const rawGaze = gazeFromBlendshapes(blendshapes);
  const eyesClosed = eyesClosedFromBlendshapes(blendshapes);

  // Apply the user's personal baseline. Roll isn't calibrated — most users
  // don't tilt their head at rest, and it doesn't factor into "looking away".
  const headPose: HeadPose = baseline
    ? {
        yaw: rawHeadPose.yaw - baseline.yaw,
        pitch: rawHeadPose.pitch - baseline.pitch,
        roll: rawHeadPose.roll,
      }
    : rawHeadPose;
  const gaze: Gaze = baseline
    ? { x: rawGaze.x - baseline.gaze.x, y: rawGaze.y - baseline.gaze.y }
    : rawGaze;

  return { ...decideStatus(headPose, gaze, eyesClosed), headPose, gaze, eyesClosed };
}

/**
 * Average a batch of samples into a single baseline. Returns null if there
 * aren't enough samples to be trustworthy.
 */
export function computeBaseline(
  samples: AttentionSample[],
  minSamples: number,
): AttentionBaseline | null {
  if (samples.length < minSamples) return null;
  let gx = 0, gy = 0, yaw = 0, pitch = 0;
  for (const s of samples) {
    gx += s.gaze.x;
    gy += s.gaze.y;
    yaw += s.headPose.yaw;
    pitch += s.headPose.pitch;
  }
  const n = samples.length;
  return {
    gaze: { x: gx / n, y: gy / n },
    yaw: yaw / n,
    pitch: pitch / n,
  };
}

function decideStatus(
  headPose: HeadPose,
  gaze: Gaze,
  eyesClosed: boolean,
): Pick<AttentionState, "status"> {
  if (eyesClosed) return { status: "eyes-closed" };
  if (
    Math.abs(headPose.yaw) > HEAD_AWAY_DEG ||
    Math.abs(headPose.pitch) > HEAD_AWAY_DEG ||
    Math.hypot(gaze.x, gaze.y) > GAZE_AWAY
  ) {
    return { status: "looking-away" };
  }
  return { status: "on-screen" };
}

function toScoreMap(cats: Category[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of cats) out[c.categoryName] = c.score;
  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
