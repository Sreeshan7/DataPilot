// Types a dataset NAME inside a sentence (no upload, no link) and captures the result. Usage: node scripts/ask-name.mjs "<message>"
import { launch, open, OUT } from "./shots.mjs";

const message = process.argv[2] ?? "could you analuze this dataset for me LIAR Dataset";
const browser = await launch();
const { page } = await open(browser, { width: 1440, height: 900 });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.locator("#composer-input").fill(message);
await page.locator("#composer-input").press("Enter");
await page.getByText("Dataset profile").waitFor({ timeout: 150000 });
await page.waitForTimeout(600);
const el = page.locator('[role="log"]');
const h = await el.evaluate((n) => n.scrollHeight);
await page.setViewportSize({ width: 1440, height: Math.min(Math.max(h + 220, 900), 3200) });
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/ask-name.png` });
console.log("matched card:", await page.locator('section[aria-label="Matched dataset"]').count(), "| console errors:", errors.length);
await browser.close();
