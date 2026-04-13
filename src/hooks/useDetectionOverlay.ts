"use client";

import { RefObject, useEffect } from "react";
import type { Detection } from "@/lib/faceApi";

const LANDMARK_LABELS = ["LE", "RE", "N", "LM", "RM"];

interface Options {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  detection: Detection | null;
}

export function useDetectionOverlay({ videoRef, canvasRef, detection }: Options) {
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detection) return;

    const sx = rect.width / video.videoWidth;
    const sy = rect.height / video.videoHeight;
    const [x1, y1, x2, y2] = detection.box;

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

    detection.landmarks.forEach(([lx, ly], i) => {
      const px = lx * sx;
      const py = ly * sy;
      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "10px monospace";
      ctx.fillText(LANDMARK_LABELS[i], px + 5, py - 3);
    });

    ctx.fillStyle = "#22c55e";
    ctx.font = "bold 12px monospace";
    ctx.fillText(`score: ${detection.score.toFixed(2)}`, x1 * sx, y1 * sy - 5);
  }, [detection, videoRef, canvasRef]);
}
