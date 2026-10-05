import { Fragment } from "react";

/** Renders `backticked` spans from backend text (column names, methods) as inline code. Plain text otherwise. */
export function Rich({ children }: { children: string }) {
  const parts = children.split(/`([^`\n]+)`/g);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1
          ? <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.88em] text-foreground">{part}</code>
          : <Fragment key={i}>{part}</Fragment>)}
    </>
  );
}
