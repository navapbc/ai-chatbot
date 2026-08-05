import { afterAll, beforeAll, beforeEach, expect, test } from 'vitest';
import { chromium, type Browser, type Page } from 'playwright';
import { executeCliCommand } from '@/evals/browser-commands';

const FIXTURE = `<!doctype html><html><body>
  <form>
    <label for="first">First Name</label>
    <input id="first" name="first" />
    <label for="state">State</label>
    <select id="state" name="state">
      <option value="">Pick</option>
      <option value="CA">California</option>
    </select>
    <input id="agree" name="agree" type="checkbox" />
    <button id="submit" type="button">Submit</button>
  </form>
</body></html>`;

let browser: Browser;
let page: Page;
let refs: Map<string, string>;

beforeAll(async () => {
  browser = await chromium.launch({ headless: true });
});

afterAll(async () => {
  await browser?.close();
});

beforeEach(async () => {
  page = await browser.newPage();
  refs = new Map();
  await page.setContent(FIXTURE);
});

test('snapshot assigns refs that later commands resolve', async () => {
  const snap = await executeCliCommand(page, refs, ['snapshot', '-i']);
  expect(snap.success).toBe(true);
  expect(snap.output).toMatch(/First Name/);

  // The ref table is what makes `@eN` usable in the next command.
  const ref = [...refs.entries()].find(([, sel]) => sel === '#first')?.[0];
  expect(ref).toBeDefined();

  const fill = await executeCliCommand(page, refs, ['fill', `@${ref}`, 'Jane']);
  expect(fill.success).toBe(true);
  expect(await page.inputValue('#first')).toBe('Jane');
});

test('unknown ref fails instead of acting on the wrong element', async () => {
  const result = await executeCliCommand(page, refs, ['click', '@e99']);
  expect(result.success).toBe(false);
  expect(result.error).toMatch(/Unknown ref/);
});

test('snapshot -s scopes to a selector', async () => {
  const result = await executeCliCommand(page, refs, [
    'snapshot',
    '-s',
    'form',
  ]);
  expect(result.success).toBe(true);
  expect(result.output).toMatch(/First Name/);
});

test('snapshot -s reports a selector that matches nothing', async () => {
  const result = await executeCliCommand(page, refs, [
    'snapshot',
    '-s',
    '#nope',
  ]);
  expect(result.success).toBe(false);
  expect(result.error).toMatch(/No element matches/);
});

test('fill and get value round-trip through CSS selectors', async () => {
  await executeCliCommand(page, refs, ['fill', '#first', "O'Brien"]);
  const got = await executeCliCommand(page, refs, ['get', 'value', '#first']);
  expect(got.success).toBe(true);
  expect(JSON.parse(got.output ?? '{}').value).toBe("O'Brien");
});

test('select sets a native dropdown by value', async () => {
  const result = await executeCliCommand(page, refs, [
    'select',
    '#state',
    'CA',
  ]);
  expect(result.success).toBe(true);
  expect(await page.inputValue('#state')).toBe('CA');
});

test('check toggles a checkbox', async () => {
  await executeCliCommand(page, refs, ['check', '#agree']);
  expect(await page.isChecked('#agree')).toBe(true);
  await executeCliCommand(page, refs, ['uncheck', '#agree']);
  expect(await page.isChecked('#agree')).toBe(false);
});

test('eval returns the script value under result, as the CLI does', async () => {
  const result = await executeCliCommand(page, refs, ['eval', '1 + 1']);
  expect(result.success).toBe(true);
  expect(JSON.parse(result.output ?? '{}').result).toBe(2);
});

test('find label fills by accessible label', async () => {
  const result = await executeCliCommand(page, refs, [
    'find',
    'label',
    'First Name',
    'fill',
    'Ada',
  ]);
  expect(result.success).toBe(true);
  expect(await page.inputValue('#first')).toBe('Ada');
});

test('wait with a number waits rather than treating it as a selector', async () => {
  const result = await executeCliCommand(page, refs, ['wait', '10']);
  expect(result.success).toBe(true);
});

test('unsupported commands fail loudly so evals cannot silently pass', async () => {
  const result = await executeCliCommand(page, refs, ['profiler', 'start']);
  expect(result.success).toBe(false);
  expect(result.error).toMatch(/Unsupported command/);
});

test('a failing command reports the error instead of throwing', async () => {
  // Shorten the auto-wait so this asserts the error path, not the full timeout.
  page.setDefaultTimeout(500);
  const result = await executeCliCommand(page, refs, ['fill', '#missing', 'x']);
  expect(result.success).toBe(false);
  expect(result.error).toBeTruthy();
});

test('type appends to an existing value; fill replaces it', async () => {
  // Regression: prompts told the model to use `type` for masked fields,
  // carried over from the pre-0.33 library where `type` took clear: true.
  // The CLI has no clear option, so `type` appends.
  await executeCliCommand(page, refs, ['fill', '#first', 'ABC']);
  await executeCliCommand(page, refs, ['type', '#first', 'XYZ']);
  expect(await page.inputValue('#first')).toBe('ABCXYZ');

  await executeCliCommand(page, refs, ['fill', '#first', 'ZZZ']);
  expect(await page.inputValue('#first')).toBe('ZZZ');
});

test('fill survives a mask that repositions the caret, type does not', async () => {
  // A mask that resets the caret to 0 on every input reverses keystroke entry
  // (92595 -> 59529, exactly what the IHSS zip field did). `fill` sets the
  // value in one operation, so it is unaffected.
  await page.setContent(`<!doctype html><html><body>
    <input id="zip" name="zip">
    <script>
      const z = document.querySelector('#zip');
      z.addEventListener('input', () => z.setSelectionRange(0, 0));
    </script>
  </body></html>`);

  await executeCliCommand(page, refs, ['type', '#zip', '92595']);
  expect(await page.inputValue('#zip')).toBe('59529');

  await executeCliCommand(page, refs, ['fill', '#zip', '']);
  await executeCliCommand(page, refs, ['fill', '#zip', '92595']);
  expect(await page.inputValue('#zip')).toBe('92595');
});
