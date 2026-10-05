import { MessageSquare, PanelLeftClose, Plus, Trash2 } from "lucide-react";
import { Wordmark } from "@/components/branding/Logo";
import type { Conversation } from "@/features/conversations/types";
import { cn } from "@/lib/utils";

function group(conversations: Conversation[]): { label: string; items: Conversation[] }[] {
  const startOfDay = new Date(); startOfDay.setHours(0, 0, 0, 0);
  const today = startOfDay.getTime();
  const week = today - 6 * 86_400_000;
  const buckets = { Today: [] as Conversation[], "Previous 7 days": [] as Conversation[], Older: [] as Conversation[] };
  for (const c of [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (c.turns.length === 0) continue;
    (c.updatedAt >= today ? buckets.Today : c.updatedAt >= week ? buckets["Previous 7 days"] : buckets.Older).push(c);
  }
  return Object.entries(buckets).filter(([, items]) => items.length).map(([label, items]) => ({ label, items }));
}

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onCollapse?: () => void;
}

export function SidebarContent({ conversations, activeId, onNew, onSelect, onDelete, onCollapse }: Props) {
  const groups = group(conversations);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-14 shrink-0 items-center justify-between px-4">
        <Wordmark />
        {onCollapse && (
          <button type="button" aria-label="Collapse sidebar" onClick={onCollapse} className="hidden size-8 place-items-center rounded-lg text-subtle transition-colors hover:bg-muted hover:text-foreground md:grid">
            <PanelLeftClose className="size-[18px]" aria-hidden />
          </button>
        )}
      </div>
      <div className="px-3 pb-2">
        <button type="button" onClick={onNew} className="flex h-10 w-full items-center gap-2.5 rounded-xl border border-border bg-raised px-3 text-sm font-medium text-foreground transition-colors hover:bg-muted">
          <Plus className="size-4 text-accent" aria-hidden /> New conversation
        </button>
      </div>
      <nav aria-label="Conversations" className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {groups.length === 0 ? (
          <p className="px-3 py-6 text-[13px] leading-5 text-subtle">Your conversations appear here. They're kept in this browser only.</p>
        ) : groups.map(({ label, items }) => (
          <div key={label} className="mt-3">
            <h2 className="px-3 pb-1 text-[11px] font-medium uppercase tracking-wider text-subtle">{label}</h2>
            <ul>
              {items.map((c) => (
                <li key={c.id} className="group relative">
                  <button
                    type="button" onClick={() => onSelect(c.id)} aria-current={c.id === activeId ? "page" : undefined}
                    className={cn("flex h-9 w-full items-center gap-2.5 rounded-lg px-3 pr-9 text-left text-[13.5px] transition-colors",
                      c.id === activeId ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground")}
                  >
                    <MessageSquare className="size-3.5 shrink-0 opacity-60" aria-hidden />
                    <span className="truncate">{c.title}</span>
                  </button>
                  <button
                    type="button" aria-label={`Delete conversation: ${c.title}`} onClick={() => onDelete(c.id)}
                    className="absolute right-1.5 top-1/2 grid size-7 -translate-y-1/2 place-items-center rounded-md text-subtle opacity-0 transition-opacity hover:bg-background hover:text-danger focus-visible:opacity-100 group-hover:opacity-100 max-md:opacity-100"
                  >
                    <Trash2 className="size-3.5" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div className="shrink-0 border-t border-border px-4 py-3 text-[11px] leading-4 text-subtle">
        History is stored in this browser, not on a server.
      </div>
    </div>
  );
}
