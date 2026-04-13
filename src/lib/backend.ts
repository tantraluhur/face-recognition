// Client for the Python FastAPI backend.
// Sends two images and gets back similarity scores from buffalo_l + antelopev2.

import { BACKEND_URL } from "@/lib/config";

const HEALTH_TIMEOUT_MS = 2000;
const VERIFY_TIMEOUT_MS = 15_000;

export interface BackendModelResult {
  similarity: number | null;
  verified: boolean | null;
  threshold: number;
  took_ms: number;
  error: string | null;
}

export interface BackendVerifyResponse {
  buffalo_l: BackendModelResult;
  antelopev2: BackendModelResult;
}

export interface BackendHealth {
  status: string;
  models: { name: string; description: string; size_mb: number }[];
  threshold: number;
}

export async function checkBackendHealth(): Promise<BackendHealth | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/health`, {
      method: "GET",
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    return (await res.json()) as BackendHealth;
  } catch {
    return null;
  }
}

export async function verifyOnBackend(
  profileImageBase64: string,
  liveImageBase64: string,
): Promise<BackendVerifyResponse> {
  const res = await fetch(`${BACKEND_URL}/api/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile_image: profileImageBase64,
      live_image: liveImageBase64,
    }),
    signal: AbortSignal.timeout(VERIFY_TIMEOUT_MS),
  });
  if (!res.ok) {
    throw new Error(`Backend returned ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as BackendVerifyResponse;
}
