import { useEffect, useState } from "react";
import { getReadiness } from "@/api/datasets";
import type { ReadyResponse } from "@/types";

export type Readiness =
  | { state: "checking" }
  | { state: "offline" }
  | { state: "online"; info: ReadyResponse };

/** Polls GET /ready so the top bar reports the backend's real state, never an assumed one. */
export function useReadiness(intervalMs = 30_000): Readiness {
  const [value, setValue] = useState<Readiness>({ state: "checking" });
  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    const check = async () => {
      try {
        const info = await getReadiness({ signal: controller.signal });
        if (alive) setValue({ state: "online", info });
      } catch {
        if (alive && !controller.signal.aborted) setValue({ state: "offline" });
      }
    };
    void check();
    const timer = window.setInterval(check, intervalMs);
    const onOnline = () => void check();
    window.addEventListener("online", onOnline);
    return () => { alive = false; controller.abort(); window.clearInterval(timer); window.removeEventListener("online", onOnline); };
  }, [intervalMs]);
  return value;
}
