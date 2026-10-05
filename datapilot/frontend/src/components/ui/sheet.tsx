import { useEffect, useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** A left-hand drawer for mobile navigation: Esc closes, focus moves in and returns, the page behind is inert. */
export function Sheet({ open, onClose, label, children }: { open: boolean; onClose: () => void; label: string; children: ReactNode }) {
  const panel = useRef<HTMLDivElement>(null);
  const returnTo = useRef<Element | null>(null);

  useEffect(() => {
    if (!open) return;
    returnTo.current = document.activeElement;
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      (returnTo.current as HTMLElement | null)?.focus?.();
    };
  }, [open, onClose]);

  return (
    <div
      className={cn("fixed inset-0 z-50 md:hidden", open ? "pointer-events-auto" : "pointer-events-none")}
      aria-hidden={!open}
      // Closed means unreachable: hidden from the tab order too, but only after the slide-out has finished.
      style={{ visibility: open ? "visible" : "hidden", transition: open ? "none" : "visibility 0s linear 250ms" }}
    >
      <div onClick={onClose} className={cn("absolute inset-0 bg-black/55 transition-opacity duration-200", open ? "opacity-100" : "opacity-0")} />
      <div
        ref={panel}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        className={cn(
          "absolute inset-y-0 left-0 w-[min(20rem,86vw)] border-r border-border bg-surface shadow-2xl outline-none transition-transform duration-250 ease-out",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {children}
      </div>
    </div>
  );
}
