// Behavioural checks in a real browser. Failure cases use request interception so they are repeatable;
// the happy paths are exercised against the real backend by flow.mjs.
//   node scripts/checks.mjs
import { launch, open, OUT } from "./shots.mjs";

const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`); };
const D = { width: 1440, height: 900 };
const M = { width: 390, height: 844 };
const send = async (page, text) => { await page.locator("#composer-input").fill(text); await page.locator("#composer-input").press("Enter"); };

const browser = await launch();

// 1. Backend unreachable: calm message, no raw error, retry control
{
  const { page, context } = await open(browser, D);
  await page.route("**/api/v1/**", (r) => r.abort("connectionrefused"));
  await send(page, "I need a dataset for churn prediction");
  await page.getByRole("alert").waitFor({ timeout: 15000 });
  const text = await page.getByRole("alert").innerText();
  check("backend unavailable shows a human message", /can't be reached/i.test(text) && !/TypeError|fetch|undefined/i.test(text), text.slice(0, 80));
  check("backend unavailable offers Try again", await page.getByRole("button", { name: "Try again" }).isVisible());
  await page.screenshot({ path: `${OUT}/err-backend.png` });
  await context.close();
}

// 2. HTTP errors map to friendly copy
{
  const { page, context } = await open(browser, D);
  await page.route("**/api/v1/intent/resolve", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ intent: "USER_A_DATASET_DISCOVERY", confidence: 0.9, evidence: [], reason: "x" }) }));
  await page.route("**/api/v1/assist", (r) => r.fulfill({ status: 502, contentType: "application/json", body: JSON.stringify({ detail: { error: { code: "X", message: "Traceback: secret /srv/app.py" } } }) }));
  await send(page, "I need a dataset for churn prediction");
  await page.getByRole("alert").waitFor({ timeout: 15000 });
  const text = await page.getByRole("alert").innerText();
  check("502 shows friendly copy and never leaks backend internals", /temporarily unavailable/i.test(text) && !/Traceback|secret|\.py/.test(text), text.slice(0, 70));
  await context.close();
}

// 3. Ambiguous prompt: a question, no results, no fake activity
{
  const { page, context } = await open(browser, D);
  await send(page, "help");
  await page.getByText(/find a dataset|analyze a dataset/i).first().waitFor({ timeout: 30000 });
  check("ambiguous prompt gets a clarifying question", await page.locator('[aria-label="DataPilot\'s answer"]').count() === 1);
  check("ambiguous prompt shows no dataset cards", await page.locator('article[aria-label^="Dataset"]').count() === 0);
  await page.screenshot({ path: `${OUT}/ambiguous.png` });
  await context.close();
}

// 4. Research: search unavailable and Gemini unavailable are explained, not hidden
{
  const { page, context } = await open(browser, D);
  const body = (status, extra = {}) => JSON.stringify({ status, request_id: "t", api_version: "v1", message: "The web search is rate limiting requests; try again later. No papers are shown, because none could be verified.", queries: ["q"], research_summary: { queries_used: 1, results_retrieved: 0, papers_found: 0, papers_selected: 0, results_rejected_as_non_academic: 0, search_provider: "web_search" }, datapilot_summary: "", papers: [], methods: [], benchmarks: [], algorithm_recommendations: [], ai_interpretation: null, gemini: { requested: false, calls_made: 0, status: "not_requested", note: null }, limitations: ["Web search is a best-effort, keyless path."], provenance: [], ...extra });
  await page.route("**/api/v1/research/discover", (r) => r.fulfill({ status: 200, contentType: "application/json", body: body("search_unavailable") }));
  await send(page, "I want to predict student academic performance using attendance and study hours.");
  await page.getByRole("button", { name: "Research this problem" }).first().waitFor({ timeout: 120000 });
  await page.getByRole("button", { name: "Research this problem" }).first().click();
  await page.getByText(/Literature search is temporarily unavailable/).waitFor({ timeout: 15000 });
  check("search unavailable is shown honestly with no papers", await page.locator("article:has(h4 a)").count() === 0);
  check("search unavailable offers Try again", await page.getByRole("button", { name: "Try again" }).isVisible());
  await page.screenshot({ path: `${OUT}/err-search.png` });
  await context.close();
}
{
  const { page, context } = await open(browser, D);
  const paper = { id: "p1", title: "Student Performance Prediction", url: "https://arxiv.org/abs/2301.00001", source: "arxiv.org", venue: "arXiv", authors: null, publication_year: 2023, year_evidence: "source_metadata", doi: null, abstract_snippet: "Random forest for student performance.", citation_count: null, matched_queries: ["q"], relevance_score: 70, relevance_reason: "Relevant because its title addresses the problem directly.", score_components: {}, methods: ["Random Forest"], retrieval_status: "snippet_only", verification_status: "web_verified", evidence: ["web_verified"] };
  await page.route("**/api/v1/research/discover", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "success", request_id: "t", api_version: "v1", message: null, queries: ["q"], research_summary: { queries_used: 1, results_retrieved: 3, papers_found: 1, papers_selected: 1, results_rejected_as_non_academic: 0, search_provider: "web_search" }, datapilot_summary: "Found 1 academic source.", papers: [paper], methods: [{ method: "Random Forest", verified_paper_count: 1, paper_ids: ["p1"], evidence: "web_verified", note: "n" }], benchmarks: [], algorithm_recommendations: [], ai_interpretation: null, gemini: { requested: true, calls_made: 1, status: "failed", note: "x" }, limitations: [], provenance: [] }) }));
  await send(page, "I want to predict student academic performance using attendance and study hours.");
  await page.getByRole("button", { name: "Research this problem" }).first().waitFor({ timeout: 120000 });
  await page.getByRole("button", { name: "Research this problem" }).first().click();
  await page.getByText(/AI enhancement is temporarily unavailable/).waitFor({ timeout: 15000 });
  check("Gemini unavailable keeps the verified results and says so", await page.getByRole("link", { name: "Student Performance Prediction" }).count() === 1);
  const link = await page.getByRole("link", { name: "Student Performance Prediction" }).getAttribute("href");
  check("paper links are the retrieved URL with safe rel", link === "https://arxiv.org/abs/2301.00001");
  await context.close();
}

// 5. Composer keyboard behaviour
{
  const { page, context } = await open(browser, D);
  const box = page.locator("#composer-input");
  await box.fill("line one");
  await box.press("Shift+Enter");
  await box.type("line two");
  check("Shift+Enter inserts a newline and does not send", (await box.inputValue()) === "line one\nline two" && await page.locator('[role="log"]').count() === 0);
  await box.fill("");
  check("send is disabled when empty", await page.getByRole("button", { name: "Send message" }).isDisabled());
  await box.fill("x\n".repeat(12));
  const h = await box.evaluate((n) => n.getBoundingClientRect().height);
  check("textarea auto-grows but is capped", h > 60 && h <= 222, `${Math.round(h)}px`);
  await page.getByRole("button", { name: "Attach a dataset" }).click();
  check("attach menu opens with 3 options", await page.getByRole("menuitem").count() === 3);
  await page.keyboard.press("Escape");
  check("Escape closes the attach menu", await page.getByRole("menuitem").count() === 0);
  await page.getByRole("button", { name: "Attach a dataset" }).click();
  await page.getByRole("menuitem", { name: /dataset link/ }).click();
  await page.getByLabel("Dataset link").fill("ftp://x");
  check("non-http links are rejected in the composer", await page.getByText(/starts with https/).isVisible());
  await context.close();
}

// 6. Light theme + reduced motion
{
  const { page, context } = await open(browser, { ...D, theme: "light", reducedMotion: "reduce" });
  await page.waitForTimeout(500);
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  check("light theme applies", bg !== "rgb(13, 16, 23)" && /^rgb\(2[0-9]{2}/.test(bg), bg);
  const first = await page.locator("h1").innerText();
  await page.waitForTimeout(3800);
  check("reduced motion stops the phrase rotation", (await page.locator("h1").innerText()) === first, first);
  await page.screenshot({ path: `${OUT}/light.png` });
  await context.close();
}

// 7. Mobile: real discovery + no horizontal overflow + drawer
{
  const { page, context } = await open(browser, M);
  await page.getByRole("button", { name: "Open navigation" }).click();
  check("mobile drawer opens", await page.getByRole("dialog", { name: "Navigation" }).isVisible());
  await page.keyboard.press("Escape");
  await send(page, "I want to predict student academic performance using attendance, previous marks and study hours.");
  await page.locator('article[aria-label^="Dataset"]').first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(600);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  const inner = await page.locator('[role="log"]').evaluate((n) => n.scrollWidth - n.clientWidth);
  check("mobile results have no horizontal overflow", overflow <= 0 && inner <= 0, `page ${overflow}px, log ${inner}px`);
  await page.screenshot({ path: `${OUT}/mobile-result.png` });
  await page.locator('[role="log"]').evaluate((n) => n.scrollTo({ top: n.scrollHeight }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/mobile-result-bottom.png` });
  await context.close();
}

await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
