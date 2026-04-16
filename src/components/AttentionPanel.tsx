import type {
  AttentionBaseline,
  AttentionState,
  AttentionStatus,
} from "@/lib/attention";

const STATUS_STYLES: Record<
  AttentionStatus,
  { label: string; pill: string; dot: string }
> = {
  "on-screen": {
    label: "On screen",
    pill: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    dot: "bg-green-500",
  },
  "looking-away": {
    label: "Looking away",
    pill: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
    dot: "bg-amber-500",
  },
  "eyes-closed": {
    label: "Eyes closed",
    pill: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
    dot: "bg-red-500",
  },
  "no-face": {
    label: "No face",
    pill: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
    dot: "bg-zinc-400",
  },
};

export function AttentionPanel({
  attention,
  ready,
  error,
  phase,
  baseline,
  onRecalibrate,
}: {
  attention: AttentionState;
  ready: boolean;
  error: string | null;
  phase: "calibrating" | "ready";
  baseline: AttentionBaseline | null;
  onRecalibrate: () => void;
}) {
  return (
    <section className="mt-8 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Attention Tracking</h2>
          <p className="text-xs text-zinc-400">
            MediaPipe Face Landmarker · head pose + gaze + eyes state
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onRecalibrate}
            disabled={!ready || phase === "calibrating"}
            className="rounded-md cursor-pointer border border-zinc-300 bg-white px-2.5 py-1 text-xs font-medium text-zinc-600 transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Recalibrate
          </button>
          <StatusPill
            status={attention.status}
            ready={ready}
            error={error}
            phase={phase}
          />
        </div>
      </div>

      {phase === "calibrating" && ready && !error && (
        <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-center text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-300">
          Calibrating — please look straight at the camera and hold still for a moment.
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <HeadPoseCard pose={attention.headPose} />
        <GazeCard gaze={attention.gaze} />
        <EyesCard eyesClosed={attention.eyesClosed} hasFace={attention.status !== "no-face"} />
      </div>

      {baseline && <BaselineReadout baseline={baseline} />}
    </section>
  );
}

function BaselineReadout({ baseline }: { baseline: AttentionBaseline }) {
  return (
    <p className="mt-3 font-mono text-[10px] text-zinc-400">
      baseline: gaze=({baseline.gaze.x.toFixed(2)}, {baseline.gaze.y.toFixed(2)}) · yaw={baseline.yaw.toFixed(0)}° · pitch={baseline.pitch.toFixed(0)}°
    </p>
  );
}

// ── Status pill (top-right) ───────────────────────────────────────────
function StatusPill({
  status,
  ready,
  error,
  phase,
}: {
  status: AttentionStatus;
  ready: boolean;
  error: string | null;
  phase: "calibrating" | "ready";
}) {
  if (error) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-700 dark:bg-red-900/30 dark:text-red-400">
        <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
        {error}
      </span>
    );
  }
  if (!ready) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1 text-xs font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
        <span className="h-1.5 w-1.5 rounded-full bg-zinc-400" />
        Loading landmarker...
      </span>
    );
  }
  if (phase === "calibrating") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
        Calibrating...
      </span>
    );
  }
  const style = STATUS_STYLES[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${style.pill}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  );
}

// ── Head pose (yaw / pitch / roll) ─────────────────────────────────────
function HeadPoseCard({ pose }: { pose: AttentionState["headPose"] }) {
  return (
    <Card title="Head pose">
      {pose ? (
        <div className="grid grid-cols-3 gap-2 font-mono text-sm">
          <PoseValue label="yaw" value={pose.yaw} hint="← / →" />
          <PoseValue label="pitch" value={pose.pitch} hint="↑ / ↓" />
          <PoseValue label="roll" value={pose.roll} hint="tilt" />
        </div>
      ) : (
        <Muted>No face</Muted>
      )}
    </Card>
  );
}

function PoseValue({
  label,
  value,
  hint,
}: {
  label: string;
  value: number;
  hint: string;
}) {
  return (
    <div className="rounded-lg bg-zinc-50 p-2 text-center dark:bg-zinc-950">
      <p className="text-[10px] uppercase tracking-wider text-zinc-400">{label}</p>
      <p className="text-base font-semibold text-zinc-700 dark:text-zinc-200">
        {value.toFixed(0)}°
      </p>
      <p className="text-[10px] text-zinc-400">{hint}</p>
    </div>
  );
}

// ── Gaze vector (2D indicator) ─────────────────────────────────────────
function GazeCard({ gaze }: { gaze: AttentionState["gaze"] }) {
  return (
    <Card title="Gaze">
      {gaze ? (
        <div className="flex items-center gap-4">
          <GazeIndicator x={gaze.x} y={gaze.y} />
          <div className="font-mono text-xs text-zinc-500">
            <p>x: {gaze.x.toFixed(2)}</p>
            <p>y: {gaze.y.toFixed(2)}</p>
          </div>
        </div>
      ) : (
        <Muted>No face</Muted>
      )}
    </Card>
  );
}

function GazeIndicator({ x, y }: { x: number; y: number }) {
  // Flip y because CSS top=0 is at the top of the screen.
  const px = 50 + x * 40; // center 50%, spread ±40%
  const py = 50 - y * 40;
  return (
    <div className="relative h-20 w-20 rounded-full border border-zinc-300 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-950">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="h-px w-full bg-zinc-200 dark:bg-zinc-800" />
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="h-full w-px bg-zinc-200 dark:bg-zinc-800" />
      </div>
      <div
        className="absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-green-500"
        style={{ left: `${px}%`, top: `${py}%` }}
      />
    </div>
  );
}

// ── Eyes state ────────────────────────────────────────────────────────
function EyesCard({
  eyesClosed,
  hasFace,
}: {
  eyesClosed: boolean;
  hasFace: boolean;
}) {
  return (
    <Card title="Eyes">
      {!hasFace ? (
        <Muted>No face</Muted>
      ) : (
        <p
          className={`text-xl font-semibold ${
            eyesClosed
              ? "text-red-600 dark:text-red-400"
              : "text-green-700 dark:text-green-400"
          }`}
        >
          {eyesClosed ? "Closed" : "Open"}
        </p>
      )}
    </Card>
  );
}

// ── Card shell ────────────────────────────────────────────────────────
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950">
      <p className="mb-3 text-xs font-bold uppercase tracking-wider text-zinc-400">{title}</p>
      {children}
    </div>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-zinc-400">{children}</p>;
}
