import { MATCH_THRESHOLD } from "@/lib/config";
import type { BackendStatus } from "@/hooks/useBackendHealth";

type Tier = "Frontend" | "Backend";

interface ModelCardProps {
  tier: Tier;
  modelName: string;
  modelInfo: string;
  similarity: number | null;
  tookMs: number | null;
  error?: string | null;
  backendStatus: BackendStatus;
}

export function ModelCard(props: ModelCardProps) {
  const { tier, backendStatus, error, similarity } = props;

  if (tier === "Backend" && backendStatus === "offline") {
    return <CardShell {...props} footer={<Muted className="text-amber-600">Backend offline</Muted>} />;
  }
  if (tier === "Backend" && backendStatus === "checking") {
    return <CardShell {...props} footer={<Muted>Checking...</Muted>} />;
  }
  if (error) {
    return <CardShell {...props} footer={<Muted className="text-red-500">{error}</Muted>} />;
  }
  if (similarity === null) {
    return <CardShell {...props} footer={<Muted>Waiting for face...</Muted>} />;
  }
  return <ResultCard {...props} similarity={similarity} />;
}

function CardShell({
  tier,
  modelName,
  modelInfo,
  footer,
}: ModelCardProps & { footer: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 text-center dark:border-zinc-800 dark:bg-zinc-950">
      <TierLabel>{tier}</TierLabel>
      <ModelHeading name={modelName} info={modelInfo} />
      <div className="mt-4">{footer}</div>
    </div>
  );
}

function ResultCard({
  tier,
  modelName,
  modelInfo,
  similarity,
  tookMs,
}: ModelCardProps & { similarity: number }) {
  const isMatch = similarity > MATCH_THRESHOLD;
  const bg = isMatch ? "bg-green-50 dark:bg-green-900/10" : "bg-red-50 dark:bg-red-900/10";
  const text = isMatch
    ? "text-green-700 dark:text-green-400"
    : "text-red-700 dark:text-red-400";

  return (
    <div className={`rounded-lg border border-zinc-200 ${bg} p-4 text-center dark:border-zinc-800`}>
      <TierLabel>{tier}</TierLabel>
      <ModelHeading name={modelName} info={modelInfo} />
      <p className={`mt-4 text-3xl font-bold ${text}`}>{(similarity * 100).toFixed(1)}%</p>
      <p className={`text-xs font-medium ${text}`}>{isMatch ? "✓ Match" : "✗ No Match"}</p>
      {tookMs !== null && <p className="mt-2 text-xs text-zinc-400">{tookMs}ms</p>}
    </div>
  );
}

function TierLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-bold uppercase tracking-wider text-zinc-400">{children}</p>
  );
}

function ModelHeading({ name, info }: { name: string; info: string }) {
  return (
    <>
      <p className="mt-1 text-sm font-semibold text-zinc-700 dark:text-zinc-200">{name}</p>
      <p className="text-xs text-zinc-400">{info}</p>
    </>
  );
}

function Muted({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <p className={`text-xs text-zinc-400 ${className}`}>{children}</p>;
}
