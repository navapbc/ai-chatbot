// Throwaway spike: Chromium with the extension, form opened, stays alive.
import { chromium } from 'playwright';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';

const ext = process.env.EXT_DIR ?? path.join(import.meta.dirname, 'extension');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'spike-profile-'));
const ctx = await chromium.launchPersistentContext(dir, {
  channel: 'chromium',
  headless: process.env.HEADED ? false : true,
  args: [`--disable-extensions-except=${ext}`, `--load-extension=${ext}`],
});
const page = await ctx.newPage();
await page.goto(process.env.FORM_URL ?? 'http://127.0.0.1:9333/form.html');
console.log('browser ready, form open');
await new Promise(() => {});
