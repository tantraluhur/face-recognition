"use client";

import { useCallback, useState } from "react";
import { detectFace, extractEmbedding } from "@/lib/faceApi";
import { MAX_PROFILE_IMAGE_DIM } from "@/lib/config";

export interface ProfileState {
  embedding: Float32Array | null;
  preview: string | null;
  aligned: string | null;
  base64: string | null;
  error: string | null;
}

const EMPTY: ProfileState = {
  embedding: null,
  preview: null,
  aligned: null,
  base64: null,
  error: null,
};

export function useProfileUpload() {
  const [state, setState] = useState<ProfileState>(EMPTY);

  const upload = useCallback(async (file: File) => {
    const previewUrl = URL.createObjectURL(file);
    setState({ ...EMPTY, preview: previewUrl });

    try {
      const canvas = await fileToCanvas(previewUrl, MAX_PROFILE_IMAGE_DIM);
      const base64 = canvas.toDataURL("image/jpeg", 0.9);

      const detection = await detectFace(canvas);
      if (!detection) {
        setState({
          ...EMPTY,
          preview: previewUrl,
          base64,
          error: "No face detected in the uploaded image. Please try another photo.",
        });
        return;
      }

      const { embedding, alignedFaceDataUrl } = await extractEmbedding(canvas, detection);
      setState({
        embedding,
        preview: previewUrl,
        aligned: alignedFaceDataUrl,
        base64,
        error: null,
      });
    } catch (err) {
      console.error("Profile upload error:", err);
      setState({ ...EMPTY, preview: previewUrl, error: "Error processing image. Please try again." });
    }
  }, []);

  return { profile: state, upload };
}

async function fileToCanvas(url: string, maxDim: number): Promise<HTMLCanvasElement> {
  const img = document.createElement("img");
  img.src = url;
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Image failed to load"));
  });

  let { width, height } = img;
  const longest = Math.max(width, height);
  if (longest > maxDim) {
    const scale = maxDim / longest;
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context unavailable");
  ctx.drawImage(img, 0, 0, width, height);
  return canvas;
}
