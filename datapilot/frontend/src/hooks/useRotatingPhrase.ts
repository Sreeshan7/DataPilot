import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "./useReducedMotion";

export const HERO_PHRASES = [
  "Let's pilot",
  "Let's discover",
  "Let's find the signal",
  "Let's map your data",
  "Let's read the evidence",
  "Let's build something",
  "Ready when you are",
] as const;

export interface RotatingPhrase {
  phrase: string;
  /** "in" while the phrase is settled or arriving, "out" during the short exit. */
  phase: "in" | "out";
}

/**
 * Cycles through phrases every `intervalMs` with a short exit before each change. With reduced motion it
 * shows one fixed phrase (no cycling at all), and `active=false` freezes it where it is.
 */
export function useRotatingPhrase(phrases: readonly string[], active = true, intervalMs = 3000, exitMs = 320): RotatingPhrase {
  const reduced = useReducedMotion();
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<"in" | "out">("in");
  const timers = useRef<number[]>([]);

  useEffect(() => {
    if (!active || reduced || phrases.length < 2) return;
    const cycle = window.setInterval(() => {
      setPhase("out");
      const swap = window.setTimeout(() => {
        setIndex((i) => (i + 1) % phrases.length);
        setPhase("in");
      }, exitMs);
      timers.current.push(swap);
    }, intervalMs);
    return () => {
      window.clearInterval(cycle);
      timers.current.forEach(window.clearTimeout);
      timers.current = [];
    };
  }, [active, reduced, phrases.length, intervalMs, exitMs]);

  return { phrase: phrases[reduced ? 0 : index], phase: reduced ? "in" : phase };
}
