import { useCallback, useState } from "react";
import { ChatWorkspace } from "@/components/chat/ChatWorkspace";
import { SidebarContent } from "@/components/chat/Sidebar";
import { TopBar } from "@/components/chat/TopBar";
import { Sheet } from "@/components/ui/sheet";
import { useConversationStore } from "@/features/conversations/store";
import { useReadiness } from "@/hooks/useReadiness";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

export function App() {
  const store = useConversationStore();
  const readiness = useReadiness();
  const { theme, toggle } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const closeDrawer = useCallback(() => setDrawer(false), []);

  const nav = {
    conversations: store.conversations,
    activeId: store.active?.id ?? null,
    onNew: () => { store.newConversation(); setDrawer(false); },
    onSelect: (id: string) => { store.select(id); setDrawer(false); },
    onDelete: store.remove,
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <a href="#composer-input" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[60] focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">Skip to the message box</a>
      <aside
        aria-label="Sidebar"
        className={cn("hidden shrink-0 overflow-hidden border-r border-border bg-surface/60 transition-[width] duration-250 ease-out md:block", collapsed ? "w-0 border-r-0" : "w-[17.5rem]")}
      >
        <div className="h-full w-[17.5rem]"><SidebarContent {...nav} onCollapse={() => setCollapsed(true)} /></div>
      </aside>
      <Sheet open={drawer} onClose={closeDrawer} label="Navigation"><SidebarContent {...nav} /></Sheet>

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar
          readiness={readiness} theme={theme} onToggleTheme={toggle}
          onOpenMobileNav={() => setDrawer(true)} sidebarCollapsed={collapsed} onExpandSidebar={() => setCollapsed(false)}
          title={store.active?.turns.length ? store.active.title : undefined}
        />
        <ChatWorkspace store={store} aiAvailable={readiness.state === "online" && readiness.info.gemini_configured} />
      </main>
    </div>
  );
}
