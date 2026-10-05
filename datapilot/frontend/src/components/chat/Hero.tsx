import { LogoMark } from "@/components/branding/Logo";
import { HERO_PHRASES, useRotatingPhrase } from "@/hooks/useRotatingPhrase";
import { cn } from "@/lib/utils";

/** The empty state: brand mark, a phrase that rotates quietly, and a prompt. No mode picker: just talk. */
export function Hero({ typing }: { typing: boolean }) {
  const { phrase, phase } = useRotatingPhrase(HERO_PHRASES, !typing);
  return (
    <div className="flex flex-col items-center text-center">
      <LogoMark className="mb-6 h-12 w-12 text-foreground" title="" />
      <h1 className="min-h-[2.75rem] text-[clamp(1.9rem,4.4vw,2.75rem)] font-semibold leading-tight tracking-tight" aria-live="off">
        <span key={phrase} className={cn("inline-block", phase === "in" ? "animate-phrase-in" : "animate-phrase-out")}>{phrase}</span>
      </h1>
      <p className="mt-3 max-w-md text-balance text-[15px] leading-6 text-muted-foreground">
        Tell me about your project or your dataset. I'll work out what you need.
      </p>
    </div>
  );
}
