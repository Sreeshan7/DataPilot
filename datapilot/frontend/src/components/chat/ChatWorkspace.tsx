import { ArrowDown } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Composer, type ComposerHandle } from "@/components/composer/Composer";
import { useAssistant, type Submission } from "@/features/assistant/useAssistant";
import type { ConversationStore } from "@/features/conversations/store";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { cn } from "@/lib/utils";
import { Hero } from "./Hero";
import { AssistantMessage, UserMessage, type MessageHandlers } from "./Messages";

const SUGGESTIONS = [
  { label: "Find a dataset", hint: "I need a dataset to predict ", action: "text" as const },
  { label: "Analyze a dataset", hint: "", action: "file" as const },
  { label: "Research a problem", hint: "I want to research the best methods for predicting ", action: "text" as const },
];

export function ChatWorkspace({ store, aiAvailable }: { store: ConversationStore; aiAvailable: boolean }) {
  const assistant = useAssistant(store);
  const composer = useRef<ComposerHandle>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const [atBottom, setAtBottom] = useState(true);
  const [typing, setTyping] = useState(false);

  const turns = store.active?.turns ?? [];
  const empty = turns.length === 0;
  const pending = turns.find((t) => t.role === "assistant" && t.status === "pending");
  const busy = Boolean(pending);

  const toBottom = useCallback((smooth = true) => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth && !reduced ? "smooth" : "auto" });
  }, [reduced]);

  // A new question is anchored near the top, so the working status and then the start of the answer are in
  // view. The page never auto-follows a growing answer: the reader decides when to scroll.
  const lastUserId = [...turns].reverse().find((t) => t.role === "user")?.id;
  useEffect(() => {
    if (!pending || !lastUserId) return;
    const el = scroller.current;
    const target = el?.querySelector<HTMLElement>(`[data-turn="${lastUserId}"]`);
    if (el && target) el.scrollTo({ top: target.getBoundingClientRect().top - el.getBoundingClientRect().top + el.scrollTop - 12, behavior: reduced ? "auto" : "smooth" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending?.id]);

  const onSubmit = useCallback((s: Submission) => { assistant.send(s); }, [assistant]);

  const handlers: MessageHandlers = {
    onAction: assistant.runAction,
    onAnalyzeUrl: (url, name) => assistant.send({ text: `Analyze ${name}`, link: url }),
    onRetry: (turnId) => store.active && assistant.retry(store.active.id, turnId),
    onCancel: assistant.cancel,
    onPickFile: () => composer.current?.openFilePicker(),
    busy, aiAvailable, fileFor: assistant.fileFor,
  };

  const composerEl = (
    <Composer
      ref={composer} busy={busy} onSubmit={onSubmit} autoFocus
      onStop={() => pending && assistant.cancel(pending.id)}
      placeholder={empty ? "Tell me about your project or your dataset…" : "Reply to DataPilot…"}
      onTypingChange={setTyping}
    />
  );

  if (empty) {
    return (
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        <div className="m-auto w-full max-w-[44rem] px-4 pb-16 pt-6 sm:px-6">
          <Hero typing={typing} />
          <div className="mt-8">{composerEl}</div>
          <div className="mt-3 flex flex-wrap justify-center gap-2" role="group" aria-label="Ways to start">
            {SUGGESTIONS.map((s) => (
              <button key={s.label} type="button"
                onClick={() => (s.action === "file" ? composer.current?.openFilePicker() : composer.current?.setText(s.hint))}
                className="rounded-full border border-border bg-surface px-3.5 py-1.5 text-[13px] text-muted-foreground transition-colors hover:border-accent/40 hover:bg-muted hover:text-foreground">
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div
        ref={scroller}
        onScroll={(e) => { const el = e.currentTarget; setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 140); }}
        className="min-h-0 flex-1 overflow-y-auto"
        role="log" aria-label="Conversation" aria-live="off"
      >
        <div className="mx-auto w-full max-w-[48rem] space-y-8 px-4 pb-24 pt-4 sm:px-6">
          {turns.map((t) => (
            <div key={t.id} data-turn={t.id}>
              {t.role === "user" ? <UserMessage turn={t} /> : <AssistantMessage turn={t} h={handlers} />}
            </div>
          ))}
        </div>
      </div>
      <div className="relative shrink-0 bg-gradient-to-t from-background via-background to-transparent px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 sm:px-6">
        <button
          type="button" aria-label="Scroll to latest" onClick={() => toBottom()}
          className={cn("absolute -top-11 left-1/2 grid size-9 -translate-x-1/2 place-items-center rounded-full border border-border bg-surface text-muted-foreground shadow-lg transition-all hover:text-foreground",
            atBottom ? "pointer-events-none translate-y-1 opacity-0" : "opacity-100")}
          tabIndex={atBottom ? -1 : 0}
        >
          <ArrowDown className="size-4" aria-hidden />
        </button>
        <div className="mx-auto w-full max-w-[48rem]">{composerEl}</div>
      </div>
    </div>
  );
}
