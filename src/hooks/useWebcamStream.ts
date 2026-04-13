"use client";

import { RefObject, useEffect, useState } from "react";

interface Options {
  videoRef: RefObject<HTMLVideoElement | null>;
  enabled: boolean;
}

export interface WebcamState {
  ready: boolean;
  error: string | null;
}

export function useWebcamStream({ videoRef, enabled }: Options): WebcamState {
  const [state, setState] = useState<WebcamState>({ ready: false, error: null });

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let stream: MediaStream | null = null;

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
        setState({ ready: true, error: null });
      })
      .catch(() => {
        if (!cancelled) setState({ ready: false, error: "Camera access denied or unavailable." });
      });

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [enabled, videoRef]);

  return state;
}
