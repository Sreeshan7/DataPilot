import { Menu, Moon, PanelLeftOpen, Sun } from "lucide-react";
import { LogoMark } from "@/components/branding/Logo";
import type { Readiness } from "@/hooks/useReadiness";
import type { Theme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

function StatusPill({ readiness }: { readiness: Readiness }) {
  const map = {
    checking: { dot: "bg-subtle animate-pulse-dot", text: "Connecting" },
    offline: { dot: "bg-danger", text: "Offline" },
    online: { dot: "bg-success", text: "Connected" },
  } as const;
  const { dot, text } = map[readiness.state];
  const ai = readiness.state === "online" ? (readiness.info.gemini_configured ? "AI summaries available" : "AI summaries not configured") : null;
  return (
    <span role="status" aria-live="polite" title={ai ?? undefined} className="inline-flex h-8 items-center gap-2 rounded-full border border-border px-3 text-xs text-muted-foreground">
      <span className={cn("size-1.5 rounded-full", dot)} aria-hidden />
      {text}
    </span>
  );
}

interface Props {
  readiness: Readiness;
  theme: Theme;
  onToggleTheme: () => void;
  onOpenMobileNav: () => void;
  sidebarCollapsed: boolean;
  onExpandSidebar: () => void;
  title?: string;
}

export function TopBar({ readiness, theme, onToggleTheme, onOpenMobileNav, sidebarCollapsed, onExpandSidebar, title }: Props) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 px-3 md:px-4">
      <button type="button" aria-label="Open navigation" onClick={onOpenMobileNav} className="grid size-9 place-items-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground md:hidden">
        <Menu className="size-5" aria-hidden />
      </button>
      {sidebarCollapsed && (
        <button type="button" aria-label="Expand sidebar" onClick={onExpandSidebar} className="hidden size-9 place-items-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground md:grid">
          <PanelLeftOpen className="size-[18px]" aria-hidden />
        </button>
      )}
      <div className="flex min-w-0 flex-1 items-center gap-2 md:hidden">
        <LogoMark className="h-6 w-6 text-foreground" title="" />
        <span className="truncate text-sm font-medium">{title ?? "DataPilot"}</span>
      </div>
      <span className="hidden min-w-0 flex-1 truncate text-sm text-muted-foreground md:block">{title}</span>
      <StatusPill readiness={readiness} />
      <button type="button" aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} onClick={onToggleTheme} className="grid size-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
        {theme === "dark" ? <Sun className="size-[18px]" aria-hidden /> : <Moon className="size-[18px]" aria-hidden />}
      </button>
    </header>
  );
}
