import { RotateCw, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";

/** A failure the user can act on. Messages are already human-readable (see api/client.ts); nothing raw reaches here. */
export function ErrorNotice({ message, retryable, onRetry }: { message: string; retryable: boolean; onRetry?: () => void }) {
  return (
    <div role="alert" className="animate-fade-up flex items-start gap-3 rounded-xl border border-danger/30 bg-danger/10 p-4">
      <TriangleAlert className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-[14px] leading-6 text-foreground">{message}</p>
        {retryable && onRetry && (
          <Button size="sm" variant="secondary" className="mt-2.5" onClick={onRetry}><RotateCw aria-hidden />Try again</Button>
        )}
      </div>
    </div>
  );
}
