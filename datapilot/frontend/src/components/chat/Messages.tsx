import { FileSpreadsheet, Link2, Paperclip, Table2 } from "lucide-react";
import { ActivityPending, ActivityTrace, ResultSkeleton } from "@/components/activity/Activity";
import { LogoMark } from "@/components/branding/Logo";
import { DiscoveryResult } from "@/components/datasets/DiscoveryResult";
import { AnalysisView } from "@/components/intelligence/AnalysisView";
import { ResearchOffer } from "@/components/research/ResearchOffer";
import { ResearchResult } from "@/components/research/ResearchResult";
import { Button } from "@/components/ui/button";
import type { ActionOutcome } from "@/features/assistant/useAssistant";
import type { AssistantTurn, UserTurn } from "@/features/conversations/types";
import { traceFromAssist, traceFromResearch, type Workflow } from "@/lib/activity";
import type { AssistResponse, NextAction } from "@/types";
import { ErrorNotice } from "./ErrorNotice";
import { NextActions, visibleActions } from "./NextActions";

export interface MessageHandlers {
  onAction: (action: NextAction, options?: { useAi?: boolean; question?: string }) => ActionOutcome;
  onAnalyzeUrl: (url: string, name: string) => void;
  onRetry: (turnId: string) => void;
  onCancel: (turnId: string) => void;
  onPickFile: () => void;
  busy: boolean;
  aiAvailable: boolean;
  fileFor: (turnId: string) => { name: string; size: number } | undefined;
}

export function UserMessage({ turn }: { turn: UserTurn }) {
  const Icon = turn.attachment?.kind === "link" ? Link2 : turn.attachment?.kind === "name" ? Table2 : FileSpreadsheet;
  return (
    <div className="flex animate-fade-up justify-end">
      <div className="max-w-[min(85%,38rem)] space-y-2.5">
        {turn.attachment && (
          <div className="ml-auto flex w-fit max-w-full items-center gap-2 rounded-xl border border-border bg-raised px-3 py-1.5 text-[13px] text-muted-foreground">
            <Icon className="size-4 shrink-0 text-accent" aria-hidden /><span className="truncate">{turn.attachment.label}</span>
          </div>
        )}
        <div className="whitespace-pre-wrap break-words rounded-2xl rounded-tr-md bg-muted px-5 py-3 text-[15px] leading-7 [overflow-wrap:anywhere]">{turn.text}</div>
      </div>
    </div>
  );
}

function Shell({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <div className="flex gap-4 sm:gap-5" role="article" aria-label={label}>
      <div className="hidden pt-1 sm:block"><LogoMark className="h-7 w-7 text-foreground" title="" /></div>
      <div className="min-w-0 flex-1 space-y-6">{children}</div>
    </div>
  );
}

function workflowOf(turn: AssistantTurn): Workflow {
  if (turn.workflow === "research") return "research";
  return turn.understood?.journey === "analysis" ? "analysis" : "discovery";
}

function Clarification({ data, onPickFile }: { data: AssistResponse; onPickFile: () => void }) {
  return (
    <div className="space-y-4">
      <p className="text-[15px] leading-7">{data.prompt_to_user ?? data.headline}</p>
      {(data.status === "NEEDS_INPUT" || data.routing.intent !== "USER_A_DATASET_DISCOVERY") && (
        <Button size="sm" variant="secondary" onClick={onPickFile}><Paperclip aria-hidden />Upload a file</Button>
      )}
    </div>
  );
}

export function AssistantMessage({ turn, h }: { turn: AssistantTurn; h: MessageHandlers }) {
  const workflow = workflowOf(turn);

  if (turn.status === "pending") {
    const file = h.fileFor(turn.id);
    return (
      <Shell label="DataPilot is working">
        <ActivityPending workflow={workflow} understood={turn.understood} progress={turn.progress} fileName={file?.name} fileSize={file?.size} onCancel={() => h.onCancel(turn.id)} />
        {turn.understood?.journey !== "ambiguous" && !turn.progress?.fraction && <ResultSkeleton workflow={workflow} />}
      </Shell>
    );
  }

  if (turn.status === "error" || !turn.result) {
    return (
      <Shell label="DataPilot could not finish">
        <ErrorNotice message={turn.error?.message ?? "That didn't work. Please try again."} retryable={turn.error?.retryable ?? true} onRetry={() => h.onRetry(turn.id)} />
      </Shell>
    );
  }

  if (turn.result.kind === "research") {
    const data = turn.result.data;
    return (
      <Shell label="Research results">
        <ResearchResult data={data} onRetry={() => h.onRetry(turn.id)} />
        <ActivityTrace steps={traceFromResearch(data)} />
        {data.status === "success" && (
          <div className="flex flex-wrap gap-2.5" role="group" aria-label="Suggested next steps">
            <Button size="sm" variant="secondary" disabled={h.busy} onClick={h.onPickFile}><Paperclip aria-hidden />Upload a dataset to analyze</Button>
          </div>
        )}
      </Shell>
    );
  }

  const data = turn.result.data;
  const research = data.next_actions.find((a) => a.action === "RESEARCH_PROBLEM");
  const hide = data.discovery ? ["ANALYZE_RECOMMENDED_DATASET", "UPLOAD_FILE"] : [];
  const actions = visibleActions(data.next_actions, hide);

  return (
    <Shell label="DataPilot's answer">
      {data.status !== "COMPLETED" ? (
        <Clarification data={data} onPickFile={h.onPickFile} />
      ) : (
        <>
          <p className="text-[15px] leading-7">{data.headline}</p>
          {data.discovery && <DiscoveryResult data={data.discovery} busy={h.busy} onAnalyze={h.onAnalyzeUrl} />}
          {data.analysis && <AnalysisView data={data.analysis} />}
        </>
      )}
      <ActivityTrace steps={traceFromAssist(data)} />
      {research && data.status === "COMPLETED" && (
        <ResearchOffer action={research} busy={h.busy} aiAvailable={h.aiAvailable} onRun={(useAi, question) => h.onAction(research, { useAi, question })} />
      )}
      <NextActions actions={actions} busy={h.busy} onAction={(a) => { if (h.onAction(a) === "needs_file") h.onPickFile(); }} />
    </Shell>
  );
}
