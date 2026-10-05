// End-to-end browser flows against the REAL backend (needs backend on :8000 and `npm run dev`).
//   node scripts/flow.mjs discovery   -> type a project, watch activity, capture the live recommendations
//   node scripts/flow.mjs analyze     -> upload a real CSV, capture the analysis
//   node scripts/flow.mjs research    -> run discovery, then click "Research this problem"
import { launch, open, OUT } from "./shots.mjs";
import { writeFileSync } from "node:fs";

const mode = process.argv[2];
const size = process.argv[3] === "mobile" ? { width: 390, height: 844 } : { width: 1440, height: 900 };
const tag = process.argv[3] === "mobile" ? "m" : "d";
const PROJECT = "I want to predict student academic performance using attendance, previous marks, assignment scores and study hours.";

async function fullShot(page, name) {
  // The conversation scrolls inside a container, so expose all of it for a single capture.
  const el = page.locator('[role="log"]');
  const h = await el.evaluate((n) => n.scrollHeight);
  await page.setViewportSize({ width: size.width, height: Math.min(Math.max(h + 220, size.height), 6000) });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}/${name}.png` });
  await page.setViewportSize(size);
}

const browser = await launch();
const { page } = await open(browser, size);
const log = [];
page.on("console", (m) => { if (m.type() === "error") log.push(m.text()); });
page.on("pageerror", (e) => log.push(String(e)));

if (mode === "discovery" || mode === "research") {
  await page.locator("#composer-input").fill(PROJECT);
  await page.locator("#composer-input").press("Enter");
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/${mode}-${tag}-pending.png` });
  await page.locator('article[aria-label^="Dataset"]').first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(800);
  await fullShot(page, `${mode}-${tag}-result`);
  console.log("cards:", await page.locator('article[aria-label^="Dataset"]').count());
}

if (mode === "research") {
  await page.getByRole("button", { name: "Research this problem" }).first().click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/research-${tag}-pending.png` });
  await page.waitForTimeout(3500);
  await page.screenshot({ path: `${OUT}/research-${tag}-pending2.png` });
  await page.locator('section[aria-label="Research summary"], p:has-text("Literature search is temporarily unavailable"), p:has-text("found no academic sources")').first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(600);
  await fullShot(page, `research-${tag}-result`);
  console.log("papers:", await page.locator('article:has(h4 a)').count());
}

if (mode === "analyze") {
  const csv = process.argv[4] ?? ".shots/churn.csv";
  await page.getByLabel("Attach a dataset").click();
  const [chooser] = await Promise.all([page.waitForEvent("filechooser"), page.getByRole("menuitem", { name: /Upload a file/ }).click()]);
  await chooser.setFiles(csv);
  await page.locator("#composer-input").fill("Here is my customer data. I want to predict which customers will churn.");
  await page.locator("#composer-input").press("Enter");
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/analyze-${tag}-pending.png` });
  await page.getByText("Dataset profile").waitFor({ timeout: 120000 });
  await page.waitForTimeout(800);
  await fullShot(page, `analyze-${tag}-result`);
}

writeFileSync(`${OUT}/${mode}-${tag}-console.txt`, log.join("\n"));
console.log("console errors:", log.length, log.slice(0, 3));
await browser.close();
