import { useId } from "react";
import { cn } from "@/lib/utils";

/**
 * The DataPilot mark: a compass ring (navigation) broken at the top right, a pilot's heading arrow
 * (piloting) and a single data node sitting in the gap (data → signal). It uses `currentColor` for the
 * arrow and the accent token for the ring, so it works on dark and light surfaces and down to 16px.
 */
export function LogoMark({ className, title = "DataPilot" }: { className?: string; title?: string }) {
  const id = useId();
  return (
    <svg viewBox="0 0 32 32" fill="none" role="img" aria-labelledby={id} className={cn("h-8 w-8 shrink-0", className)}>
      <title id={id}>{title}</title>
      <path d="M26.2 17.6A10.5 10.5 0 1 1 14.4 5.8" stroke="hsl(var(--accent))" strokeWidth="2" strokeLinecap="round" />
      <path d="M22.8 8.4 18 22.6l-2.6-5.9-5.9-2.6 13.3-5.7Z" fill="currentColor" />
      <circle cx="24" cy="12" r="1.9" fill="hsl(var(--accent))" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-2.5 text-foreground", className)}>
      <LogoMark />
      <span className="text-[17px] font-semibold tracking-tight">
        Data<span className="text-accent">Pilot</span>
      </span>
    </span>
  );
}
