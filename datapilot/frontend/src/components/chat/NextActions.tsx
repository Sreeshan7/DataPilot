import { ArrowUpRight, Paperclip, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { NextAction } from "@/types";

const LABEL: Record<string, string> = {
  UPLOAD_FILE: "Upload a file to analyze",
  FIND_ALTERNATIVE_DATASET: "Find a better dataset",
};

export function actionLabel(a: NextAction): string {
  return a.label ?? LABEL[a.action] ?? a.description;
}

/** The backend's own suggestions as buttons. Research is rendered separately (it has its own explicit card). */
export function visibleActions(actions: NextAction[], hide: string[] = []): NextAction[] {
  const seen = new Set<string>();
  return actions.filter((a) => {
    if (a.action === "RESEARCH_PROBLEM" || hide.includes(a.action)) return false;
    const key = `${a.action}:${JSON.stringify(a.payload)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function NextActions({ actions, onAction, busy }: { actions: NextAction[]; onAction: (a: NextAction) => void; busy: boolean }) {
  if (actions.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="Suggested next steps">
      {actions.slice(0, 5).map((a, i) => (
        <Button key={`${a.action}-${i}`} size="sm" variant="secondary" disabled={busy} title={a.description} onClick={() => onAction(a)} className="max-w-full">
          {a.action === "UPLOAD_FILE" ? <Paperclip aria-hidden /> : a.action === "FIND_ALTERNATIVE_DATASET" ? <Search aria-hidden /> : <ArrowUpRight aria-hidden />}
          <span className="truncate">{actionLabel(a)}</span>
        </Button>
      ))}
    </div>
  );
}
