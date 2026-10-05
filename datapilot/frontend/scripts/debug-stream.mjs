// Prints what the working card shows over time (to check live progress reaches the page). Usage: node scripts/debug-stream.mjs
import { launch, open } from "./shots.mjs";

const browser = await launch();
const { page } = await open(browser, { width: 1440, height: 900 });
page.on("console", (m) => { if (m.type() === "error") console.log("console error:", m.text()); });
await page.locator("#composer-input").fill("I want to predict student academic performance using attendance, previous marks and study hours.");
await page.locator("#composer-input").press("Enter");
for (let i = 0; i < 8; i++) {
  await page.waitForTimeout(700);
  const text = await page.locator('section[aria-label="DataPilot is working"]').innerText().catch(() => "(card gone)");
  console.log(`--- ${(i + 1) * 0.7}s`, text.replace(/\n+/g, " | ").slice(0, 400));
}
await browser.close();
