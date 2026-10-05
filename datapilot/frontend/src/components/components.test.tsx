import { act, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Composer } from "@/components/composer/Composer";
import { Markdown } from "@/components/chat/Markdown";
import { Rich } from "@/components/chat/Rich";
import { PaperCard } from "@/components/research/ResearchResult";
import { ResearchOffer } from "@/components/research/ResearchOffer";
import { HERO_PHRASES, useRotatingPhrase } from "@/hooks/useRotatingPhrase";
import type { NextAction, Paper } from "@/types";

describe("Composer", () => {
  it("Enter sends, Shift+Enter inserts a newline", async () => {
    const onSubmit = vi.fn();
    render(<Composer busy={false} onSubmit={onSubmit} />);
    const box = screen.getByLabelText("Message DataPilot");
    const user = userEvent.setup();
    await user.type(box, "hello{Shift>}{Enter}{/Shift}world");
    expect(onSubmit).not.toHaveBeenCalled();
    expect((box as HTMLTextAreaElement).value).toBe("hello\nworld");
    await user.keyboard("{Enter}");
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ text: "hello\nworld" }));
    expect((box as HTMLTextAreaElement).value).toBe("");
  });
  it("does not send an empty message and shows Stop while busy", async () => {
    const onSubmit = vi.fn();
    const { rerender } = render(<Composer busy={false} onSubmit={onSubmit} />);
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    rerender(<Composer busy onSubmit={onSubmit} />);
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    await userEvent.setup().type(screen.getByLabelText("Message DataPilot"), "x{Enter}");
    expect(onSubmit).not.toHaveBeenCalled();
  });
  it("reports whether the user has started typing", async () => {
    const onTypingChange = vi.fn();
    render(<Composer busy={false} onSubmit={() => undefined} onTypingChange={onTypingChange} />);
    await userEvent.setup().type(screen.getByLabelText("Message DataPilot"), "a");
    expect(onTypingChange).toHaveBeenLastCalledWith(true);
  });
});

describe("useRotatingPhrase", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());
  it("cycles through phrases and loops", () => {
    const { result } = renderHook(() => useRotatingPhrase(HERO_PHRASES, true, 3000, 300));
    const seen = [result.current.phrase];
    for (let i = 0; i < HERO_PHRASES.length; i++) {
      act(() => { vi.advanceTimersByTime(3000); });
      act(() => { vi.advanceTimersByTime(300); });
      seen.push(result.current.phrase);
    }
    expect(new Set(seen).size).toBe(HERO_PHRASES.length);
    expect(seen[seen.length - 1]).toBe(HERO_PHRASES[0]);
  });
  it("freezes when inactive (the user is typing)", () => {
    const { result } = renderHook(() => useRotatingPhrase(HERO_PHRASES, false, 3000, 300));
    const first = result.current.phrase;
    act(() => { vi.advanceTimersByTime(20_000); });
    expect(result.current.phrase).toBe(first);
  });
  it("has between six and ten phrases", () => {
    expect(HERO_PHRASES.length).toBeGreaterThanOrEqual(6);
    expect(HERO_PHRASES.length).toBeLessThanOrEqual(10);
  });
});

describe("untrusted text is rendered safely", () => {
  it("never renders raw HTML or javascript: links from markdown", () => {
    const { container } = render(<Markdown>{'<img src=x onerror="alert(1)"> [click](javascript:alert(1)) [ok](https://arxiv.org/abs/1)'}</Markdown>);
    expect(container.querySelector("img")).toBeNull();
    const links = [...container.querySelectorAll("a")];
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["https://arxiv.org/abs/1"]);
    expect(links[0].getAttribute("rel")).toMatch(/noopener/);
  });
  it("renders backticked spans as inline code", () => {
    const { container } = render(<Rich>{"Predict `churn` from `tenure`"}</Rich>);
    expect([...container.querySelectorAll("code")].map((c) => c.textContent)).toEqual(["churn", "tenure"]);
  });
});

const paper = (over: Partial<Paper> = {}): Paper => ({
  id: "p", title: "A Study", url: "https://arxiv.org/abs/2301.00001", source: "arxiv.org", venue: "arXiv", authors: null,
  publication_year: null, year_evidence: "unknown", doi: null, abstract_snippet: null, citation_count: null, matched_queries: [],
  relevance_score: 71.4, relevance_reason: "Relevant because x.", score_components: {}, methods: [], retrieval_status: "title_only",
  verification_status: "web_verified", evidence: ["web_verified"], ...over,
} as Paper);

describe("PaperCard never fabricates metadata", () => {
  it("shows no year or authors when the backend has none", () => {
    render(<PaperCard paper={paper()} index={0} />);
    expect(screen.queryByText(/20\d\d/)).toBeNull();
    expect(screen.getByRole("link", { name: "A Study" })).toHaveAttribute("href", "https://arxiv.org/abs/2301.00001");
    expect(screen.queryByText(/Retrieved text/)).toBeNull();
  });
  it("shows year, authors and methods only when provided, and refuses unsafe URLs", () => {
    render(<PaperCard paper={paper({ publication_year: 2024, authors: ["A. Author"], methods: ["XGBoost"], url: "javascript:alert(1)", abstract_snippet: "snip" })} index={0} />);
    expect(screen.getByText("2024")).toBeInTheDocument();
    expect(screen.getByText("A. Author")).toBeInTheDocument();
    expect(screen.getByText("XGBoost")).toBeInTheDocument();
    expect(screen.queryByRole("link")).toBeNull();
  });
});

describe("ResearchOffer", () => {
  const action = (over: Partial<NextAction> = {}): NextAction => ({ action: "RESEARCH_PROBLEM", description: "d", endpoint: "/e", payload: {}, id: "research_problem", label: "Research this problem", available: true, reason: "ok", ...over });
  it("runs only on click, with AI off by default", async () => {
    const onRun = vi.fn();
    render(<ResearchOffer action={action()} onRun={onRun} aiAvailable busy={false} />);
    expect(onRun).not.toHaveBeenCalled();
    await userEvent.setup().click(screen.getByRole("button", { name: "Research this problem" }));
    expect(onRun).toHaveBeenCalledWith(false);
  });
  it("passes the AI choice through only when the user ticks it and a key exists", async () => {
    const onRun = vi.fn();
    const user = userEvent.setup();
    render(<ResearchOffer action={action()} onRun={onRun} aiAvailable busy={false} />);
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Research this problem" }));
    expect(onRun).toHaveBeenCalledWith(true);
  });
  it("disables the AI option when no key is configured", () => {
    render(<ResearchOffer action={action()} onRun={() => undefined} aiAvailable={false} busy={false} />);
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByText(/not configured/)).toBeInTheDocument();
  });
  it("explains why research is unavailable instead of offering a dead button", () => {
    render(<ResearchOffer action={action({ available: false, reason: "Describe the problem first." })} onRun={() => undefined} aiAvailable busy={false} />);
    expect(screen.getByText("Describe the problem first.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Research this problem" })).toBeNull();
  });
});
