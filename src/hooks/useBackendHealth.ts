"use client";

import { useCallback, useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/backend";

export type BackendStatus = "checking" | "online" | "offline";

export function useBackendHealth() {
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    let cancelled = false;
    checkBackendHealth().then((health) => {
      if (!cancelled) setStatus(health ? "online" : "offline");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const markOffline = useCallback(() => setStatus("offline"), []);

  return { status, markOffline };
}
