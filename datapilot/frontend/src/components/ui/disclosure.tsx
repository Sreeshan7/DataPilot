import { ChevronDown } from "lucide-react";
import { useId, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** An accessible show/hide section (button + region, aria-expanded). Used for progressive disclosure. */
export function Disclosure({
  label, children, defaultOpen = false, className, meta,
}: { label: ReactNode; children: ReactNode; defaultOpen?: boolean; className?: string; meta?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <div className={className}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className="group -mx-1 flex w-[calc(100%+0.5rem)] items-center gap-2 rounded-md px-1 py-1 text-left text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronDown className={cn("size-4 shrink-0 transition-transform duration-200", open ? "rotate-0" : "-rotate-90")} aria-hidden />
        <span className="min-w-0 flex-1 truncate">{label}</span>
        {meta ? <span className="shrink-0 text-xs font-normal text-subtle">{meta}</span> : null}
      </button>
      <div id={id} role="region" hidden={!open} className={cn(open && "animate-fade-in pt-2")}>
        {open ? children : null}
      </div>
    </div>
  );
}
