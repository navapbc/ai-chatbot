import { readFileSync } from 'node:fs';
import path from 'node:path';
import { type Browser, type CDPSession, chromium, type Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  createMaskSession,
  createScreenshotQueue,
  isSensitiveParts,
  type MaskSession,
  PAGE_SOURCE,
  SCREENSHOT_REFUSED,
  ScreenshotMaskError,
  withScreenshotMask,
} from '../../lib/relay/screenshot-mask';
import {
  type FieldDescriptor,
  isSensitiveField,
} from '../../lib/relay/sensitive-fields';

// The sequencer (withScreenshotMask, the queue) is tested with fake
// sessions. The page side (PAGE_SOURCE driven by createMaskSession) runs in a
// real headless Chromium, because the injected source is a string the worker
// sends over CDP and only a real page proves it works.

interface Row {
  id: string;
  field: FieldDescriptor;
  sensitive: boolean;
}

const DATASET = JSON.parse(
  readFileSync(
    path.resolve(__dirname, '../../evals/datasets/sensitive-fields.json'),
    'utf8',
  ),
) as Row[];

/** A fake session that records the order of its calls into `log`. */
function fakeSession(
  name: string,
  log: string[],
  opts: { applyFails?: boolean; restoreFails?: boolean } = {},
): MaskSession {
  return {
    async apply() {
      log.push(`apply ${name}`);
      if (opts.applyFails) throw new Error('apply failed');
    },
    async restore() {
      log.push(`restore ${name}`);
      if (opts.restoreFails) throw new Error('restore failed 900-00-1234');
    },
  };
}

describe('withScreenshotMask', () => {
  it('applies before capture and restores after', async () => {
    const log: string[] = [];
    const result = await withScreenshotMask(
      [fakeSession('tab', log), fakeSession('child', log)],
      async () => {
        log.push('capture');
        return 'png';
      },
    );
    expect(result).toBe('png');
    expect(log).toEqual([
      'apply tab',
      'apply child',
      'capture',
      'restore tab',
      'restore child',
    ]);
  });

  it('restores when capture throws', async () => {
    const log: string[] = [];
    await expect(
      withScreenshotMask([fakeSession('tab', log)], async () => {
        log.push('capture');
        throw new Error('capture failed');
      }),
    ).rejects.toThrow('capture failed');
    expect(log).toEqual(['apply tab', 'capture', 'restore tab']);
  });

  it('restores the already-masked sessions and never captures when a later apply fails', async () => {
    const log: string[] = [];
    const capture = vi.fn(async () => 'png');
    const outcome = withScreenshotMask(
      [
        fakeSession('tab', log),
        fakeSession('child', log, { applyFails: true }),
        fakeSession('other', log),
      ],
      capture,
    );
    await expect(outcome).rejects.toBeInstanceOf(ScreenshotMaskError);
    await expect(outcome).rejects.toThrow(SCREENSHOT_REFUSED);
    expect(capture).not.toHaveBeenCalled();
    // The failing session is restored too (it may have masked part of its
    // page before failing); the one never applied is left alone.
    expect(log).toEqual([
      'apply tab',
      'apply child',
      'restore tab',
      'restore child',
    ]);
  });

  it('refuses with a fixed message that carries no page data', async () => {
    const outcome = withScreenshotMask(
      [
        {
          apply: async () => {
            throw new Error('field 900-00-1234 not found');
          },
          restore: async () => {},
        },
      ],
      async () => 'png',
    );
    await expect(outcome).rejects.toThrow(
      'The screenshot was refused because sensitive fields could not be masked.',
    );
    await expect(outcome).rejects.not.toThrow(/900/);
  });

  it('a restore failure is reported and does not hide the capture error', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const log: string[] = [];
      await expect(
        withScreenshotMask(
          [fakeSession('tab', log, { restoreFails: true })],
          async () => {
            throw new Error('capture failed');
          },
        ),
      ).rejects.toThrow('capture failed');
      expect(warn).toHaveBeenCalledTimes(1);
      // Reported without the restore error's text (it can quote the page).
      expect(JSON.stringify(warn.mock.calls)).not.toContain('900');
    } finally {
      warn.mockRestore();
    }
  });

  it('a throwing restore still rejects/reports without masking the capture error', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const log: string[] = [];
      const capture = new Error('capture failed');
      const outcome = withScreenshotMask(
        [
          fakeSession('tab', log, { restoreFails: true }),
          fakeSession('child', log, { restoreFails: true }),
        ],
        async () => {
          throw capture;
        },
      );
      await expect(outcome).rejects.toBe(capture);
      // Every session is restored even though the first restore threw.
      expect(log).toEqual([
        'apply tab',
        'apply child',
        'restore tab',
        'restore child',
      ]);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(JSON.stringify(warn.mock.calls)).not.toContain('900');

      // A capture that succeeded is returned: the bridge verifies the masks
      // inside `capture`, so a restore failure afterwards leaks nothing.
      warn.mockClear();
      await expect(
        withScreenshotMask(
          [fakeSession('tab', [], { restoreFails: true })],
          async () => 'png',
        ),
      ).resolves.toBe('png');
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });

  it('restore runs after a navigation error', async () => {
    const log: string[] = [];
    await expect(
      withScreenshotMask([fakeSession('tab', log)], async () => {
        log.push('capture');
        throw new Error('Cannot find context with specified id');
      }),
    ).rejects.toThrow('Cannot find context');
    expect(log).toEqual(['apply tab', 'capture', 'restore tab']);
  });

  it('a restore that throws synchronously still lets the others run', async () => {
    const log: string[] = [];
    const throwsSync: MaskSession = {
      apply: async () => {},
      restore: () => {
        throw new Error('sync');
      },
    };
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await withScreenshotMask(
        [throwsSync, fakeSession('child', log)],
        async () => 'png',
      );
      expect(log).toContain('restore child');
    } finally {
      warn.mockRestore();
    }
  });
});

describe('createScreenshotQueue', () => {
  it('two concurrent screenshots are serialized', async () => {
    const queue = createScreenshotQueue();
    const log: string[] = [];
    let releaseFirst = () => {};
    const first = queue(async () => {
      log.push('first start');
      await new Promise<void>((resolve) => {
        releaseFirst = resolve;
      });
      log.push('first end');
      throw new Error('first failed');
    });
    const second = queue(async () => {
      log.push('second start');
      return 'second';
    });
    await new Promise((r) => setTimeout(r, 20));
    expect(log).toEqual(['first start']);
    releaseFirst();
    await expect(first).rejects.toThrow('first failed');
    // A failed screenshot does not wedge the queue.
    await expect(second).resolves.toBe('second');
    expect(log).toEqual(['first start', 'first end', 'second start']);
  });
});

// --- the page side, in a real browser ------------------------------------------

let browser: Browser;
let page: Page;
let cdp: CDPSession;

const send = (method: string, params?: Record<string, unknown>) =>
  cdp.send(method as any, params as any) as Promise<unknown>;

const styles = () =>
  page.evaluate(() =>
    Array.from(document.querySelectorAll('[id]'), (el) => [
      el.id,
      el.getAttribute('style'),
    ]),
  );

/** Rendered text colour of an element, as the compositor will paint it. */
const fill = (selector: string) =>
  page.evaluate(
    (s) =>
      getComputedStyle(document.querySelector(s) as Element)
        .webkitTextFillColor,
    selector,
  );

const FORM = `<!doctype html><html lang="en"><body>
  <label for="ssn">Social Security Number</label>
  <input id="ssn" value="900-00-1234" style="width: 14em; color: rgb(20, 20, 20) !important">
  <label for="city">City</label> <input id="city" value="Springfield">
  <label>PIN <input id="pin" value="9999"></label>
  <span id="cardlbl">Card number</span>
  <div id="card" contenteditable aria-labelledby="cardlbl">4000 0000 0000 0002</div>
  <textarea id="notes">ordinary notes</textarea>
  <select id="sid" aria-label="State ID number"><option>D1234567</option></select>
  <iframe id="same" srcdoc="<label for=p>Passport number</label><input id=p value=X1234567><input id=q value=ok>"></iframe>
</body></html>`;

beforeAll(async () => {
  browser = await chromium.launch({ channel: 'chromium', headless: true });
  page = await browser.newPage();
  cdp = await page.context().newCDPSession(page);
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe('page side', { timeout: 30_000 }, () => {
  it('the injected classifier agrees with isSensitiveField on every dataset row', async () => {
    await page.setContent(
      '<!doctype html><html lang="en"><body></body></html>',
    );
    // Each row rendered twice: label as aria-label, and as a <label for>.
    // Neutral ids (x0, y0): the id joins the name signal, so a dataset id
    // such as "ssn-..." would flip the result.
    await page.evaluate((rows) => {
      rows.forEach((row, i) => {
        for (const viaLabelElement of [false, true]) {
          const input = document.createElement('input');
          const id = `${viaLabelElement ? 'y' : 'x'}${i}`;
          input.id = id;
          const f = row.field;
          if (f.inputType !== undefined)
            input.setAttribute('type', f.inputType);
          if (f.autocomplete !== undefined) {
            input.setAttribute('autocomplete', f.autocomplete);
          }
          if (f.name !== undefined) input.setAttribute('name', f.name);
          if (f.placeholder !== undefined) {
            input.setAttribute('placeholder', f.placeholder);
          }
          if (f.description !== undefined)
            input.setAttribute('title', f.description);
          if (f.role !== undefined) input.setAttribute('role', f.role);
          if (f.label !== undefined) {
            if (viaLabelElement) {
              const label = document.createElement('label');
              label.htmlFor = id;
              label.textContent = f.label;
              document.body.append(label);
            } else {
              input.setAttribute('aria-label', f.label);
            }
          }
          document.body.append(input);
        }
      });
    }, DATASET);
    const report = (await page.evaluate(`(${PAGE_SOURCE}).report()`)) as {
      fields: unknown[];
      opaque: number;
    };
    expect(report.fields).toHaveLength(DATASET.length * 2);
    const disagreements = DATASET.flatMap((row, i) => {
      const expected = isSensitiveField(row.field);
      return [report.fields[2 * i], report.fields[2 * i + 1]].some(
        (parts) => isSensitiveParts(parts) !== expected,
      )
        ? [row.id]
        : [];
    });
    expect(disagreements).toEqual([]);
    // And the dataset's own labels hold.
    expect(
      DATASET.every((row) => isSensitiveField(row.field) === row.sensitive),
    ).toBe(true);
  });

  it('masks sensitive fields (incl. same-origin frames) and restores exactly', async () => {
    await page.setContent(FORM);
    await page.frameLocator('#same').locator('#q').waitFor();
    const before = await styles();
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: () => false,
    });
    await session.apply();
    expect({
      ssn: await fill('#ssn'),
      pin: await fill('#pin'),
      card: await fill('#card'),
      sid: await fill('#sid'),
      city: await fill('#city'),
      notes: await fill('#notes'),
    }).toEqual({
      ssn: 'rgba(0, 0, 0, 0)',
      pin: 'rgba(0, 0, 0, 0)',
      card: 'rgba(0, 0, 0, 0)',
      sid: 'rgba(0, 0, 0, 0)',
      city: 'rgb(0, 0, 0)',
      notes: 'rgb(0, 0, 0)',
    });
    const inFrame = await page
      .frameLocator('#same')
      .locator('#p')
      .evaluate((el) => getComputedStyle(el).webkitTextFillColor);
    expect(inFrame).toBe('rgba(0, 0, 0, 0)');
    await session.settle();
    await session.verify();
    await session.restore();
    expect(await styles()).toEqual(before);
    expect(
      await page
        .frameLocator('#same')
        .locator('#p')
        .evaluate((el) => el.getAttribute('style')),
    ).toBeNull();
    // Restore twice is a no-op.
    await session.restore();
    expect(await styles()).toEqual(before);
  });

  it('verify refuses when the page stripped a mask during the window', async () => {
    await page.setContent(FORM);
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: () => false,
    });
    await session.apply();
    await page.evaluate(() =>
      document.getElementById('pin')?.removeAttribute('style'),
    );
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
    expect(
      await page.evaluate(() =>
        document.getElementById('pin')?.getAttribute('style'),
      ),
    ).toBeNull();
  });

  it('verify refuses after a navigation, and restore is then a no-op', async () => {
    await page.setContent(FORM);
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: () => false,
    });
    await session.apply();
    await page.goto('about:blank');
    await page.setContent(FORM);
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    // The holder died with the old document: restore cannot reach it (and
    // says so), and the new document is left untouched.
    await session.restore().catch(() => {});
    expect(await fill('#ssn')).toBe('rgb(20, 20, 20)');
    expect(
      await page.evaluate(() =>
        document.getElementById('pin')?.getAttribute('style'),
      ),
    ).toBeNull();
  });

  it('verify refuses when the document was rewritten in place (document.open)', async () => {
    await page.setContent(FORM);
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: () => false,
    });
    await session.apply();
    // setContent rewrites the same document: the holder survives, but every
    // masked element is gone and the new ones are unmasked.
    await page.setContent(FORM);
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
  });

  it('refuses when a session reports an uninspectable iframe', async () => {
    await page.setContent(
      '<!doctype html><html lang="en"><body><iframe id="opaque" sandbox srcdoc="<input id=s aria-label=SSN value=1>"></iframe></body></html>',
    );
    const covered: string[] = [];
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: (frameId) => {
        covered.push(frameId);
        return false;
      },
    });
    await expect(session.apply()).rejects.toThrow();
    // The frame was resolved to its CDP frame id before being refused.
    expect(covered).toHaveLength(1);
    expect(typeof covered[0]).toBe('string');
    await session.restore();

    // The same frame, covered by a child session, is accepted.
    const ok = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: (frameId) => frameId === covered[0],
    });
    await ok.apply();
    await ok.restore();
  });

  it('an invisible uninspectable iframe paints nothing and is not a failure', async () => {
    await page.setContent(
      '<!doctype html><html lang="en"><body><iframe sandbox style="display:none" srcdoc="x"></iframe><iframe sandbox width="0" height="0" srcdoc="x"></iframe></body></html>',
    );
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: () => false,
    });
    await session.apply();
    await session.restore();
  });

  it('a page that breaks the injected function fails apply (exceptionDetails is not success)', async () => {
    await page.setContent(FORM);
    await page.evaluate(() => {
      Document.prototype.querySelectorAll = () => {
        throw new Error('nope');
      };
    });
    const session = createMaskSession({
      send,
      objectGroup: 'mask-test',
      isCoveredFrame: () => false,
    });
    await expect(session.apply()).rejects.toThrow();
    await session.restore();
  });
});

describe('createMaskSession with a stub send', () => {
  it('rejects on a malformed reply instead of treating it as masked', async () => {
    const session = createMaskSession({
      send: async () => ({}),
      objectGroup: 'g',
      isCoveredFrame: () => true,
    });
    await expect(session.apply()).rejects.toThrow();
  });

  it('restore without a successful apply sends nothing; verify refuses', async () => {
    const stub = vi.fn(async () => ({}));
    const session = createMaskSession({
      send: stub,
      objectGroup: 'g',
      isCoveredFrame: () => true,
    });
    await session.restore();
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await expect(session.settle()).rejects.toBeInstanceOf(ScreenshotMaskError);
    expect(stub).not.toHaveBeenCalled();
  });

  it('an apply the bridge gave up on stops before masking once restore runs', async () => {
    const methods: string[] = [];
    let releaseReport = () => {};
    const send = async (method: string, params?: Record<string, unknown>) => {
      methods.push(
        method === 'Runtime.callFunctionOn'
          ? String(params?.functionDeclaration)
          : method,
      );
      if (method === 'Runtime.evaluate') return { result: { objectId: 'h' } };
      if (String(params?.functionDeclaration).includes('report')) {
        await new Promise<void>((resolve) => {
          releaseReport = resolve;
        });
        return { result: { value: { fields: [{}], opaque: 0 } } };
      }
      return { result: { value: true } };
    };
    const session = createMaskSession({
      send,
      objectGroup: 'g',
      isCoveredFrame: () => true,
    });
    const applying = session.apply();
    await new Promise((r) => setTimeout(r, 10));
    await session.restore();
    releaseReport();
    await expect(applying).rejects.toThrow();
    expect(methods.some((m) => m.includes('mask('))).toBe(false);
  });
});
