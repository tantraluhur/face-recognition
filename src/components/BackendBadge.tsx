import type { BackendStatus } from "@/hooks/useBackendHealth";

const STYLES: Record<BackendStatus, { pill: string; dot: string; label: string }> = {
  checking: {
    pill: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
    dot: "bg-zinc-400",
    label: "Backend: checking...",
  },
  online: {
    pill: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    dot: "bg-green-500",
    label: "Backend: online",
  },
  offline: {
    pill: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
    dot: "bg-amber-500",
    label: "Backend: offline (run python/main.py)",
  },
};

export function BackendBadge({ status }: { status: BackendStatus }) {
  const style = STYLES[status];
  return (
    <div className="flex justify-center">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${style.pill}`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
        {style.label}
      </span>
    </div>
  );
}
