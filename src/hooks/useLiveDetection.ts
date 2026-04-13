"use client";

import { RefObject, useEffect, useRef, useState } from "react";
import { cosineSimilarity, detectFace, extractEmbedding, type Detection } from "@/lib/faceApi";
import { verifyOnBackend, type BackendVerifyResponse } from "@/lib/backend";
import { DETECTION_INTERVAL_MS, MATCH_THRESHOLD } from "@/lib/config";
import type { BackendStatus } from "@/hooks/useBackendHealth";

export type MatchStatus = "no-profile" | "no-face" | "match" | "no-match";

export interface SnapSize {
  w: number;
  h: number;
  bytes: number;
}

export interface LiveDetectionState {
  detection: Detection | null;
  aligned: string | null;
  snapSize: SnapSize | null;
  status: string;
  matchStatus: MatchStatus;
  similarity: number | null;
  backendResults: BackendVerifyResponse | null;
}

interface Options {
  videoRef: RefObject<HTMLVideoElement | null>;
  enabled: boolean;
  profileEmbedding: Float32Array | null;
  profileBase64: string | null;
  backendStatus: BackendStatus;
  onBackendError: () => void;
}

const INITIAL: LiveDetectionState = {
  detection: null,
  aligned: null,
  snapSize: null,
  status: "Waiting...",
  matchStatus: "no-profile",
  similarity: null,
  backendResults: null,
};

export function useLiveDetection(opts: Options): LiveDetectionState {
  const { videoRef, enabled, profileEmbedding, profileBase64, backendStatus, onBackendError } = opts;
  const [state, setState] = useState<LiveDetectionState>(INITIAL);
  const liveEmbeddingRef = useRef<Float32Array | null>(null);
  const backendInFlightRef = useRef(false);

  // Re-compare when profile changes without waiting for the next detection tick.
  useEffect(() => {
    setState((prev) => ({
      ...prev,
      ...computeMatch(profileEmbedding, liveEmbeddingRef.current),
    }));
  }, [profileEmbedding]);

  useEffect(() => {
    if (!enabled) return;

    const tick = async () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      const snap = snapshotVideo(video);
      if (!snap) return;

      try {
        const detection = await detectFace(snap);
        if (!detection) {
          liveEmbeddingRef.current = null;
          setState((prev) => ({
            ...prev,
            detection: null,
            aligned: null,
            snapSize: null,
            status: "No face detected in webcam",
            similarity: null,
            matchStatus: prev.matchStatus === "no-profile" ? "no-profile" : "no-face",
          }));
          return;
        }

        const { embedding, alignedFaceDataUrl } = await extractEmbedding(snap, detection);
        liveEmbeddingRef.current = embedding;

        const liveBase64 = snap.toDataURL("image/jpeg", 0.9);
        const snapSize = measureSnap(snap, liveBase64);

        setState((prev) => ({
          ...prev,
          detection,
          aligned: alignedFaceDataUrl,
          status: "Face detected",
          snapSize,
          ...computeMatch(profileEmbedding, embedding),
        }));

        await maybeCallBackend({
          profileBase64,
          liveBase64,
          backendStatus,
          inFlightRef: backendInFlightRef,
          onResult: (backendResults) => setState((prev) => ({ ...prev, backendResults })),
          onError: onBackendError,
        });
      } catch (err) {
        console.error("Detection error:", err);
      }
    };

    const id = setInterval(tick, DETECTION_INTERVAL_MS);
    return () => clearInterval(id);
  }, [enabled, videoRef, profileEmbedding, profileBase64, backendStatus, onBackendError]);

  return state;
}

function computeMatch(
  profile: Float32Array | null,
  live: Float32Array | null,
): Pick<LiveDetectionState, "matchStatus" | "similarity"> {
  if (!profile) return { matchStatus: "no-profile", similarity: null };
  if (!live) return { matchStatus: "no-face", similarity: null };
  const similarity = cosineSimilarity(profile, live);
  return {
    matchStatus: similarity > MATCH_THRESHOLD ? "match" : "no-match",
    similarity,
  };
}

function snapshotVideo(video: HTMLVideoElement): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  // Mirror horizontally to match the on-screen (selfie-style) preview.
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  return canvas;
}

function measureSnap(canvas: HTMLCanvasElement, dataUrl: string): SnapSize {
  const payload = dataUrl.split(",")[1] ?? "";
  return {
    w: canvas.width,
    h: canvas.height,
    bytes: Math.floor((payload.length * 3) / 4),
  };
}

async function maybeCallBackend({
  profileBase64,
  liveBase64,
  backendStatus,
  inFlightRef,
  onResult,
  onError,
}: {
  profileBase64: string | null;
  liveBase64: string;
  backendStatus: BackendStatus;
  inFlightRef: { current: boolean };
  onResult: (r: BackendVerifyResponse) => void;
  onError: () => void;
}) {
  if (!profileBase64 || backendStatus !== "online" || inFlightRef.current) return;
  inFlightRef.current = true;
  try {
    const result = await verifyOnBackend(profileBase64, liveBase64);
    onResult(result);
  } catch (err) {
    console.error("Backend error:", err);
    onError();
  } finally {
    inFlightRef.current = false;
  }
}
