import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { safeUrl } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * Markdown for assistant prose. Raw HTML is never rendered (react-markdown's default) and only http(s) links
 * become links, so text that came from the web or from a model cannot inject markup or `javascript:` URLs.
 */
export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <div className={cn("prose-dp min-w-0 break-words", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: c }) => {
            const ok = safeUrl(href);
            return ok ? <a href={ok} target="_blank" rel="noopener noreferrer nofollow">{c}</a> : <span>{c}</span>;
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
