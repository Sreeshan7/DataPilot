import { ArrowUp, FileSpreadsheet, Link2, Paperclip, Plus, Square, Table2, X } from "lucide-react";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { cn } from "@/lib/utils";
import { bytes } from "@/lib/format";
import type { Submission } from "@/features/assistant/useAssistant";

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const ACCEPT = ".csv,.tsv,.xlsx";
const EXT = /\.(csv|tsv|xlsx)$/i;

export function validateFile(file: File): string | null {
  if (!EXT.test(file.name)) return "DataPilot reads .csv, .tsv and .xlsx files.";
  if (file.size > MAX_UPLOAD_BYTES) return `That file is ${bytes(file.size)}; the limit is ${bytes(MAX_UPLOAD_BYTES)}.`;
  if (file.size === 0) return "That file is empty.";
  return null;
}

/** http(s) only: the backend allowlists registries, but the form should not even accept other schemes. */
export function validateLink(value: string): string | null {
  try {
    const u = new URL(value.trim());
    return u.protocol === "https:" || u.protocol === "http:" ? null : "Use a link that starts with https://";
  } catch {
    return "That doesn't look like a link.";
  }
}

export interface ComposerHandle {
  focus: () => void;
  openFilePicker: () => void;
  setText: (value: string) => void;
}

interface Props {
  onSubmit: (s: Submission) => void;
  onStop?: () => void;
  busy: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  /** Reports whether there is text in the box, so the landing phrase can stop rotating once the user starts. */
  onTypingChange?: (typing: boolean) => void;
  className?: string;
}

type Mode = "link" | "name" | null;

export const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  { onSubmit, onStop, busy, placeholder = "Describe your project, or attach a dataset…", autoFocus, className, onTypingChange }, ref,
) {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [link, setLink] = useState("");
  const [name, setName] = useState("");
  const [mode, setMode] = useState<Mode>(null);
  const [menu, setMenu] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const composing = useRef(false);

  useImperativeHandle(ref, () => ({
    focus: () => area.current?.focus(),
    openFilePicker: () => picker.current?.click(),
    setText: (value: string) => { setText(value); window.requestAnimationFrame(() => area.current?.focus()); },
  }), []);

  useLayoutEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  useEffect(() => { if (autoFocus) area.current?.focus(); }, [autoFocus]);
  useEffect(() => { onTypingChange?.(text.length > 0); }, [text, onTypingChange]);

  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent) => { if (!menuRef.current?.contains(e.target as Node)) setMenu(false); };
    const esc = (e: globalThis.KeyboardEvent) => { if (e.key === "Escape") { setMenu(false); area.current?.focus(); } };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [menu]);

  const take = useCallback((f: File | undefined) => {
    if (!f) return;
    const problem = validateFile(f);
    setError(problem);
    if (!problem) { setFile(f); setMode(null); area.current?.focus(); }
  }, []);

  const linkProblem = mode === "link" && link.trim() ? validateLink(link) : null;
  const hasInput = Boolean(text.trim() || file || (mode === "link" && link.trim() && !linkProblem) || (mode === "name" && name.trim()));
  const canSend = hasInput && !busy;

  function submit() {
    if (!canSend) return;
    onSubmit({
      text: text.trim(), file: file ?? undefined,
      link: mode === "link" && link.trim() && !linkProblem ? link.trim() : undefined,
      datasetName: mode === "name" && name.trim() ? name.trim() : undefined,
    });
    setText(""); setFile(null); setLink(""); setName(""); setMode(null); setError(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !composing.current && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  const chip = "inline-flex max-w-full items-center gap-2 rounded-lg border border-border bg-raised py-1 pl-2.5 pr-1.5 text-[13px] text-foreground";

  return (
    <div className={className}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); take(e.dataTransfer.files?.[0]); }}
        className={cn(
          "relative rounded-2xl border bg-surface shadow-[0_1px_0_hsl(var(--border)/0.6),0_12px_32px_-12px_rgb(0_0_0/0.45)] transition-[border-color,box-shadow] duration-200",
          "focus-within:border-accent/50 focus-within:shadow-[0_0_0_3px_hsl(var(--accent)/0.12),0_12px_32px_-12px_rgb(0_0_0/0.45)]",
          dragging ? "border-accent/70 bg-accent-soft/30" : "border-border",
        )}
      >
        {(file || mode) && (
          <div className="flex flex-wrap items-center gap-2 px-3 pt-3">
            {file && (
              <span className={chip}>
                <FileSpreadsheet className="size-4 text-accent" aria-hidden />
                <span className="truncate">{file.name}</span>
                <span className="text-xs text-subtle">{bytes(file.size)}</span>
                <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFile(null)} className="rounded p-1 text-subtle hover:bg-muted hover:text-foreground"><X className="size-3.5" /></button>
              </span>
            )}
            {mode && (
              <span className={cn(chip, "w-full max-w-md")}>
                {mode === "link" ? <Link2 className="size-4 shrink-0 text-accent" aria-hidden /> : <Table2 className="size-4 shrink-0 text-accent" aria-hidden />}
                <input
                  autoFocus
                  aria-label={mode === "link" ? "Dataset link" : "Dataset name"}
                  value={mode === "link" ? link : name}
                  onChange={(e) => (mode === "link" ? setLink(e.target.value) : setName(e.target.value))}
                  onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); if (hasInput) submit(); } }}
                  placeholder={mode === "link" ? "https://www.kaggle.com/datasets/…" : "e.g. Titanic, Iris, Heart Disease"}
                  className="min-w-0 flex-1 bg-transparent py-0.5 text-[13px] outline-none placeholder:text-subtle"
                />
                <button type="button" aria-label="Remove dataset reference" onClick={() => { setMode(null); setLink(""); setName(""); }} className="rounded p-1 text-subtle hover:bg-muted hover:text-foreground"><X className="size-3.5" /></button>
              </span>
            )}
          </div>
        )}

        <label htmlFor="composer-input" className="sr-only">Message DataPilot</label>
        <textarea
          id="composer-input"
          ref={area}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          onCompositionStart={() => { composing.current = true; }}
          onCompositionEnd={() => { composing.current = false; }}
          placeholder={placeholder}
          maxLength={5000}
          className="block max-h-[220px] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[15px] leading-6 outline-none placeholder:text-subtle"
        />

        <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5 pt-1">
          <div className="relative" ref={menuRef}>
            <button
              type="button" aria-label="Attach a dataset" aria-haspopup="menu" aria-expanded={menu}
              onClick={() => setMenu((v) => !v)}
              className="grid size-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Plus className={cn("size-[18px] transition-transform duration-200", menu && "rotate-45")} aria-hidden />
            </button>
            {menu && (
              <div role="menu" aria-label="Attach" className="absolute bottom-11 left-0 z-20 w-60 animate-fade-up rounded-xl border border-border bg-surface p-1.5 shadow-2xl">
                {[
                  { icon: Paperclip, label: "Upload a file", hint: "CSV, TSV or Excel", run: () => picker.current?.click() },
                  { icon: Link2, label: "Add a dataset link", hint: "Kaggle, Hugging Face, UCI, OpenML", run: () => setMode("link") },
                  { icon: Table2, label: "Name a dataset", hint: "DataPilot looks it up", run: () => setMode("name") },
                ].map(({ icon: Icon, label, hint, run }) => (
                  <button key={label} role="menuitem" type="button" onClick={() => { setMenu(false); run(); }}
                    className="flex w-full items-start gap-3 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-muted">
                    <Icon className="mt-0.5 size-4 text-muted-foreground" aria-hidden />
                    <span><span className="block text-sm text-foreground">{label}</span><span className="block text-xs text-subtle">{hint}</span></span>
                  </button>
                ))}
              </div>
            )}
            <input ref={picker} type="file" accept={ACCEPT} hidden onChange={(e) => { take(e.target.files?.[0]); e.target.value = ""; }} />
          </div>

          {busy ? (
            <button type="button" aria-label="Stop" onClick={onStop} className="grid size-9 place-items-center rounded-full border border-border bg-raised text-foreground transition-colors hover:bg-muted">
              <Square className="size-3.5 fill-current" aria-hidden />
            </button>
          ) : (
            <button
              type="button" aria-label="Send message" disabled={!canSend} onClick={submit}
              className="grid size-9 place-items-center rounded-full bg-accent text-accent-foreground transition-[opacity,transform] duration-150 hover:scale-[1.04] disabled:scale-100 disabled:opacity-30"
            >
              <ArrowUp className="size-[18px]" aria-hidden />
            </button>
          )}
        </div>
      </div>
      <div className="min-h-5 px-1 pt-1.5 text-xs" role="status" aria-live="polite">
        {error || linkProblem ? <span className="text-danger">{error ?? linkProblem}</span> : null}
      </div>
    </div>
  );
});
