// Visual/behaviour checks against the running app (Edge via playwright-core; no browser download).
//   node scripts/shots.mjs landing        -> screenshots of the empty state at desktop and mobile sizes
// Output goes to the directory in SHOTS_DIR (default ./.shots, git-ignored).
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const EDGE = process.env.EDGE_PATH ?? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const URL = process.env.APP_URL ?? "http://localhost:5173/";
export const OUT = process.env.SHOTS_DIR ?? ".shots";
mkdirSync(OUT, { recursive: true });

export async function launch() {
  return chromium.launch({ executablePath: EDGE, headless: true });
}

export async function open(browser, { width, height, theme = "dark", reducedMotion = "no-preference" }) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, reducedMotion });
  await context.addInitScript((t) => { try { localStorage.setItem("datapilot.theme", t); } catch {} }, theme);
  const page = await context.newPage();
  await page.goto(URL, { waitUntil: "networkidle" });
  return { page, context };
}

if (process.argv[2] === "landing") {
  const browser = await launch();
  for (const [name, size] of Object.entries({ desktop: { width: 1440, height: 900 }, mobile: { width: 390, height: 844 } })) {
    const { page, context } = await open(browser, size);
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${OUT}/landing-${name}-1.png` });
    await page.waitForTimeout(3600);
    await page.screenshot({ path: `${OUT}/landing-${name}-2.png` });
    console.log(name, await page.locator("h1").innerText());
    await context.close();
  }
  await browser.close();
}
