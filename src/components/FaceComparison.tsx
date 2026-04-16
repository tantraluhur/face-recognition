"use client";

import Image from "next/image";
import { useRef } from "react";
import { AttentionPanel } from "@/components/AttentionPanel";
import { BackendBadge } from "@/components/BackendBadge";
import { ModelCard } from "@/components/ModelCard";
import { useAttentionTracking } from "@/hooks/useAttentionTracking";
import { useBackendHealth } from "@/hooks/useBackendHealth";
import { useDetectionOverlay } from "@/hooks/useDetectionOverlay";
import { useFaceModels } from "@/hooks/useFaceModels";
import { useLiveDetection } from "@/hooks/useLiveDetection";
import { useProfileUpload } from "@/hooks/useProfileUpload";
import { useWebcamStream } from "@/hooks/useWebcamStream";

export default function FaceComparison() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const models = useFaceModels();
  const backend = useBackendHealth();
  const { profile, upload } = useProfileUpload();

  const webcam = useWebcamStream({ videoRef, enabled: models.loaded });
  const live = useLiveDetection({
    videoRef,
    enabled: models.loaded && webcam.ready,
    profileEmbedding: profile.embedding,
    profileBase64: profile.base64,
    backendStatus: backend.status,
    onBackendError: backend.markOffline,
  });

  useDetectionOverlay({ videoRef, canvasRef: overlayRef, detection: live.detection });

  const attention = useAttentionTracking({
    videoRef,
    enabled: models.loaded && webcam.ready,
  });

  if (models.error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-red-500">{models.error}</p>
      </div>
    );
  }

  if (!models.loaded) {
    return <LoadingScreen />;
  }

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-8">
      <header className="mb-6">
        <h1 className="mb-2 text-center text-3xl font-bold tracking-tight">Face Comparison</h1>
        <p className="mb-2 text-center text-sm text-zinc-400">
          Frontend (MobileFaceNet) vs Backend (buffalo_l + antelopev2)
        </p>
        <BackendBadge status={backend.status} />
      </header>

      <div className="grid gap-8 md:grid-cols-2">
        <ProfilePanel
          preview={profile.preview}
          error={profile.error}
          hasEmbedding={profile.embedding !== null}
          onFileSelect={(file) => void upload(file)}
        />
        <WebcamPanel
          videoRef={videoRef}
          overlayRef={overlayRef}
          error={webcam.error}
          status={live.status}
          snapSize={live.snapSize}
        />
      </div>

      <AttentionPanel
        attention={attention.state}
        ready={attention.ready}
        error={attention.error}
        phase={attention.phase}
        baseline={attention.baseline}
        onRecalibrate={attention.recalibrate}
      />

      <AlignedFacesPanel profile={profile.aligned} live={live.aligned} />

      <ComparisonPanel
        hasProfile={profile.embedding !== null}
        frontendSimilarity={live.similarity}
        backendStatus={backend.status}
        backendResults={live.backendResults}
      />
    </div>
  );
}

// ── Presentational sections ───────────────────────────────────────────

function LoadingScreen() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-zinc-800" />
        <p className="text-zinc-500">Loading face detection models...</p>
        <p className="mt-1 text-xs text-zinc-400">First load downloads ~15MB of model data</p>
      </div>
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      {children}
    </section>
  );
}

function ProfilePanel({
  preview,
  error,
  hasEmbedding,
  onFileSelect,
}: {
  preview: string | null;
  error: string | null;
  hasEmbedding: boolean;
  onFileSelect: (file: File) => void;
}) {
  return (
    <Card>
      <h2 className="mb-4 text-lg font-semibold">Profile Photo</h2>

      <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-zinc-300 bg-zinc-50 p-6 transition hover:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:border-zinc-600">
        <UploadIcon />
        <span className="text-sm text-zinc-500">Click to upload a photo</span>
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFileSelect(file);
          }}
        />
      </label>

      {preview && (
        <div className="mt-4">
          {/* Preview uses an object URL, so next/image is not a good fit here. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Profile preview"
            className="mx-auto max-h-48 rounded-lg object-contain"
          />
        </div>
      )}

      {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      {hasEmbedding && (
        <p className="mt-3 text-sm text-green-600">
          Face embedding extracted (512-dim ArcFace vector).
        </p>
      )}
    </Card>
  );
}

function WebcamPanel({
  videoRef,
  overlayRef,
  error,
  status,
  snapSize,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  overlayRef: React.RefObject<HTMLCanvasElement | null>;
  error: string | null;
  status: string;
  snapSize: { w: number; h: number; bytes: number } | null;
}) {
  return (
    <Card>
      <h2 className="mb-4 text-lg font-semibold">Live Webcam</h2>

      <div className="relative overflow-hidden rounded-lg bg-zinc-900">
        {error ? (
          <div className="flex h-60 items-center justify-center">
            <p className="px-4 text-center text-sm text-red-400">{error}</p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="h-60 w-full -scale-x-100 object-cover"
            />
            <canvas
              ref={overlayRef}
              className="pointer-events-none absolute inset-0 h-60 w-full"
            />
          </>
        )}
      </div>

      <p className="mt-3 text-sm text-zinc-500">{status}</p>
      {snapSize && (
        <p className="mt-1 font-mono text-xs text-zinc-400">
          snap: {snapSize.w}×{snapSize.h} · JPEG ~{(snapSize.bytes / 1024).toFixed(1)} KB
        </p>
      )}
    </Card>
  );
}

function AlignedFacesPanel({
  profile,
  live,
}: {
  profile: string | null;
  live: string | null;
}) {
  if (!profile && !live) return null;

  return (
    <section className="mt-8 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <h2 className="mb-1 text-center text-lg font-semibold">Aligned Faces (112x112)</h2>
      <p className="mb-4 text-center text-xs text-zinc-400">
        These are the exact images sent to the frontend ArcFace model
      </p>
      <div className="flex items-center justify-center gap-8">
        <AlignedFace label="Profile" src={profile} />
        <div className="text-2xl text-zinc-300">vs</div>
        <AlignedFace label="Webcam" src={live} />
      </div>
    </section>
  );
}

function AlignedFace({ label, src }: { label: string; src: string | null }) {
  return (
    <div className="text-center">
      <p className="mb-2 text-xs font-medium text-zinc-500">{label}</p>
      {src ? (
        <Image
          src={src}
          alt={`Aligned ${label.toLowerCase()} face`}
          width={112}
          height={112}
          unoptimized
          className="h-28 w-28 rounded-lg border border-zinc-200 object-contain dark:border-zinc-700"
        />
      ) : (
        <div className="flex h-28 w-28 items-center justify-center rounded-lg border border-dashed border-zinc-300 dark:border-zinc-700">
          <span className="text-xs text-zinc-400">None</span>
        </div>
      )}
    </div>
  );
}

function ComparisonPanel({
  hasProfile,
  frontendSimilarity,
  backendStatus,
  backendResults,
}: {
  hasProfile: boolean;
  frontendSimilarity: number | null;
  backendStatus: ReturnType<typeof useBackendHealth>["status"];
  backendResults: ReturnType<typeof useLiveDetection>["backendResults"];
}) {
  return (
    <section className="mt-8 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <h2 className="mb-1 text-center text-lg font-semibold">Model Comparison</h2>
      <p className="mb-6 text-center text-xs text-zinc-400">
        Same input, three different models. Higher percentage = more confidently the same person.
      </p>

      {!hasProfile ? (
        <p className="text-center text-zinc-400">Upload a profile photo to begin comparison.</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-3">
          <ModelCard
            tier="Frontend"
            modelName="MobileFaceNet"
            modelInfo="13 MB · ONNX Runtime Web"
            similarity={frontendSimilarity}
            tookMs={null}
            backendStatus="online"
          />
          <ModelCard
            tier="Backend"
            modelName="buffalo_l (ResNet50)"
            modelInfo="182 MB · InsightFace"
            similarity={backendResults?.buffalo_l.similarity ?? null}
            tookMs={backendResults?.buffalo_l.took_ms ?? null}
            error={backendResults?.buffalo_l.error ?? null}
            backendStatus={backendStatus}
          />
          <ModelCard
            tier="Backend"
            modelName="antelopev2 (ResNet100)"
            modelInfo="264 MB · InsightFace"
            similarity={backendResults?.antelopev2.similarity ?? null}
            tookMs={backendResults?.antelopev2.took_ms ?? null}
            error={backendResults?.antelopev2.error ?? null}
            backendStatus={backendStatus}
          />
        </div>
      )}
    </section>
  );
}

function UploadIcon() {
  return (
    <svg className="mb-2 h-8 w-8 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  );
}
