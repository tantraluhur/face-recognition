"use client";

import { RefObject, useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import {
  attentionFromResult,
  computeBaseline,
  gazeFromBlendshapes,
  headPoseFromMatrix,
  NO_FACE,
  type AttentionBaseline,
  type AttentionSample,
  type AttentionState,
} from "@/lib/attention";

// MediaPipe publishes its WASM runtime on jsDelivr. Pinned to the same minor
// version as the npm package so the JS and WASM never drift.
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

const TICK_MS = 250;             // 4 FPS is plenty for an attention indicator
const CALIBRATION_MS = 1500;     // how long to sample the user's resting position
const MIN_CALIBRATION_SAMPLES = 4;

type CalibrationPhase = "calibrating" | "ready";

// Module-level singleton. React StrictMode double-mounts effects in dev,
// which would otherwise load the landmarker twice and then `close()` the
// first instance mid-initialization — that close call is what MediaPipe
// emits the "Console Error" for. Loading once for the whole page lifetime
// sidesteps the issue entirely.
let sharedLandmarker: Promise<FaceLandmarker> | null = null;

function getLandmarker(): Promise<FaceLandmarker> {
  if (sharedLandmarker) return sharedLandmarker;
  sharedLandmarker = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  })().catch((err) => {
    // Failure shouldn't cache — let the next mount retry.
    sharedLandmarker = null;
    throw err;
  });
  return sharedLandmarker;
}

interface Options {
  videoRef: RefObject<HTMLVideoElement | null>;
  enabled: boolean;
}

export interface AttentionTracking {
  state: AttentionState;
  ready: boolean;
  error: string | null;
  phase: CalibrationPhase;
  baseline: AttentionBaseline | null;
  recalibrate: () => void;
}

export function useAttentionTracking({ videoRef, enabled }: Options): AttentionTracking {
  const [state, setState] = useState<AttentionState>(NO_FACE);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<CalibrationPhase>("calibrating");
  const [baseline, setBaseline] = useState<AttentionBaseline | null>(null);

  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const samplesRef = useRef<AttentionSample[]>([]);
  const calibrationStartRef = useRef<number>(0);

  // ── Attach to the shared Face Landmarker ─────────────────────────────
  useEffect(() => {
    let cancelled = false;
    getLandmarker().then(
      (landmarker) => {
        if (cancelled) return;
        landmarkerRef.current = landmarker;
        setReady(true);
      },
      (err) => {
        console.error("Face Landmarker load error:", err);
        if (!cancelled) setError("Failed to load MediaPipe Face Landmarker.");
      },
    );
    return () => {
      cancelled = true;
      // The landmarker is shared across the page lifetime — we only drop
      // our reference to it, we don't close it.
      landmarkerRef.current = null;
    };
  }, []);

  // `startCalibration` is only called from event handlers (e.g. Recalibrate
  // button). Initial calibration happens automatically — initial `phase` is
  // `"calibrating"` and `calibrationStartRef.current === 0` tells the tick
  // loop to stamp the start time on its first run.
  const startCalibration = useCallback(() => {
    samplesRef.current = [];
    calibrationStartRef.current = 0;
    setBaseline(null);
    setPhase("calibrating");
  }, []);

  // ── Main detection tick ──────────────────────────────────────────────
  useEffect(() => {
    if (!ready || !enabled) return;

    const tick = () => {
      const landmarker = landmarkerRef.current;
      const video = videoRef.current;
      if (!landmarker || !video || video.readyState < 2) return;

      try {
        const result = landmarker.detectForVideo(video, performance.now());
        const matrix = result.facialTransformationMatrixes[0];
        const blendshapes = result.faceBlendshapes[0];

        // Collect calibration samples when a face is present.
        if (phase === "calibrating" && matrix && blendshapes) {
          // Lazily stamp the start time on the first useful tick. This avoids
          // calling setState from an effect just to record `performance.now()`.
          if (calibrationStartRef.current === 0) {
            calibrationStartRef.current = performance.now();
          }

          samplesRef.current.push({
            gaze: gazeFromBlendshapes(blendshapes),
            headPose: headPoseFromMatrix(matrix),
          });

          const elapsed = performance.now() - calibrationStartRef.current;
          if (
            elapsed >= CALIBRATION_MS &&
            samplesRef.current.length >= MIN_CALIBRATION_SAMPLES
          ) {
            const b = computeBaseline(samplesRef.current, MIN_CALIBRATION_SAMPLES);
            if (b) {
              setBaseline(b);
              setPhase("ready");
            }
          }
        }

        setState(attentionFromResult({ matrix, blendshapes, baseline }));
      } catch (err) {
        console.error("Face Landmarker detect error:", err);
      }
    };

    const id = setInterval(tick, TICK_MS);
    return () => clearInterval(id);
  }, [ready, enabled, videoRef, phase, baseline]);

  return { state, ready, error, phase, baseline, recalibrate: startCalibration };
}
