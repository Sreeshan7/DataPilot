import { cn } from "@/lib/utils";

/** A loading placeholder. It is a shape only: it never stands in for content that does not exist yet. */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("skeleton", className)} />;
}
