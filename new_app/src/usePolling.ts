import { useEffect, useState } from "react";
import { ApiEnvelope } from "./api";

export type LoadState<T> = {
  envelope?: ApiEnvelope<T>;
  loading: boolean;
  error?: string;
  refresh: () => void;
};

export function usePolling<T>(
  key: string,
  loader: () => Promise<ApiEnvelope<T>>,
  intervalMs: number,
  enabled = true
): LoadState<T> {
  const [envelope, setEnvelope] = useState<ApiEnvelope<T>>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    if (!enabled) {
      return;
    }
    setLoading((current) => current || envelope === undefined);
    loader()
      .then((result) => {
        if (!cancelled) {
          setEnvelope(result);
          setError(undefined);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
    // key and tick intentionally drive reloads; loader is recreated by caller with current filters.
  }, [key, tick, enabled]);

  useEffect(() => {
    if (!enabled || intervalMs <= 0) {
      return;
    }
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        setTick((value) => value + 1);
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [enabled, intervalMs]);

  return {
    envelope,
    loading,
    error,
    refresh: () => setTick((value) => value + 1)
  };
}
