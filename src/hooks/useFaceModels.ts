"use client";

import { useEffect, useState } from "react";
import { loadModels } from "@/lib/faceApi";

export interface FaceModelsState {
  loaded: boolean;
  error: string | null;
}

export function useFaceModels(): FaceModelsState {
  const [state, setState] = useState<FaceModelsState>({ loaded: false, error: null });

  useEffect(() => {
    let cancelled = false;
    loadModels()
      .then(() => {
        if (!cancelled) setState({ loaded: true, error: null });
      })
      .catch((err) => {
        console.error("Model load error:", err);
        if (!cancelled) {
          setState({ loaded: false, error: "Failed to load face detection models." });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
