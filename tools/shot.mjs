// Headless screenshot helper for the visual review loop.
//   node tools/shot.mjs out.png "t=12&cam=0,7,0,20,-4" [frames] [width] [height]
// Requires a static server on http://localhost:8000 (python3 -m http.server 8000).
// Uses SwiftShader WebGPU, so it is slow but deterministic.
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); } catch { playwright = require('/opt/node22/lib/node_modules/playwright'); }

const [out = 'shot.png', params = '', frames = '4', w = '960', h = '540'] = process.argv.slice(2);
const browser = await playwright.chromium.launch({
  headless: true, channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'],
});
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
await page.goto(`http://localhost:8000/index.html?scale=1&${params}`);
const t0 = Date.now();
await page.waitForFunction((n) => (window.__frames || 0) >= n, +frames, { timeout: 600000, polling: 500 });
await page.screenshot({ path: out, timeout: 180000 });
console.log(`frames=${await page.evaluate(() => window.__frames)} in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
const status = await page.evaluate(() => document.getElementById('status').textContent);
if (status) console.log('status:', status);
for (const l of logs) console.log(l);
await browser.close();
