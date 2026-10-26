import { readFileSync } from 'node:fs';
import path from 'node:path';
import { type Browser, type CDPSession, chromium, type Page } from 'playwright';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { hasSensitiveShape } from '../../lib/relay/redactor';
import {
  createMaskSession,
  createScreenshotQueue,
  isSensitiveParts,
  type MaskSession,
  type MaskSessionOptions,
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
  cdp.send(method as never, params as never) as Promise<unknown>;

const KNOWN = '900-77-4321';

const maskSession = (overrides: Partial<MaskSessionOptions> = {}) =>
  createMaskSession({
    send,
    objectGroup: 'mask-test',
    isCoveredFrame: () => false,
    isSensitiveValue: (v) => hasSensitiveShape(v) || v.includes(KNOWN),
    hasKnownValue: (t) => t.includes(KNOWN),
    ...overrides,
  });

const styles = () =>
  page.evaluate(() =>
    Array.from(document.querySelectorAll('[id]'), (el) => [
      el.id,
      el.getAttribute('style'),
      el.getAttributeNames().join(' '),
    ]),
  );

const TRANSPARENT = 'rgba(0, 0, 0, 0)';

/** Rendered text colour of an element (or one of its pseudo-elements). */
const fill = (selector: string, pseudo?: string) =>
  page.evaluate(
    ([s, p]) =>
      getComputedStyle(document.querySelector(s) as Element, p ?? undefined)
        .webkitTextFillColor,
    [selector, pseudo ?? null] as const,
  );

const FORM = `<!doctype html><html lang="en"><body>
  <label for="ssn">Social Security Number</label>
  <input id="ssn" value="900-00-1234" style="width: 14em; color: rgb(20, 20, 20) !important">
  <label for="city">City</label> <input id="city" value="Springfield">
  <label>PIN <input id="pin" value="9999"></label>
  <span id="cardlbl">Card number</span>
  <div id="card" contenteditable aria-labelledby="cardlbl">4000 <b>0000</b> 0000 0002</div>
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

const load = async (html: string) => {
  await page.goto('about:blank');
  await page.setContent(html);
};

describe('page side', { timeout: 30_000 }, () => {
  it('the injected classifier agrees with isSensitiveField on every dataset row', async () => {
    await load('<!doctype html><html lang="en"><body></body></html>');
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
      fields: Array<{ parts: unknown }>;
    };
    expect(report.fields).toHaveLength(DATASET.length * 2);
    const disagreements = DATASET.flatMap((row, i) => {
      const expected = isSensitiveField(row.field);
      return [report.fields[2 * i], report.fields[2 * i + 1]].some(
        (field) => isSensitiveParts(field.parts) !== expected,
      )
        ? [row.id]
        : [];
    });
    expect(disagreements).toEqual([]);
    expect(
      DATASET.every((row) => isSensitiveField(row.field) === row.sensitive),
    ).toBe(true);
  });

  it('masks sensitive fields (incl. same-origin frames, in isolated worlds) and restores exactly', async () => {
    await load(FORM);
    await page.frameLocator('#same').locator('#q').waitFor();
    const before = await styles();
    const session = maskSession();
    await session.apply();
    expect({
      ssn: await fill('#ssn'),
      pin: await fill('#pin'),
      card: await fill('#card'),
      cardChild: await fill('#card b'),
      sid: await fill('#sid'),
      city: await fill('#city'),
      notes: await fill('#notes'),
    }).toEqual({
      ssn: TRANSPARENT,
      pin: TRANSPARENT,
      card: TRANSPARENT,
      cardChild: TRANSPARENT,
      sid: TRANSPARENT,
      city: 'rgb(0, 0, 0)',
      notes: 'rgb(0, 0, 0)',
    });
    const frame = page.frameLocator('#same');
    expect(
      await frame
        .locator('#p')
        .evaluate((el) => getComputedStyle(el).webkitTextFillColor),
    ).toBe(TRANSPARENT);
    await session.settle();
    await session.verify();
    await session.restore();
    // Style attributes back exactly, and no marker attribute left behind.
    expect(await styles()).toEqual(before);
    expect(await page.evaluate(() => document.adoptedStyleSheets.length)).toBe(
      0,
    );
    expect(
      await frame.locator('#p').evaluate((el) => el.getAttributeNames()),
    ).toEqual(['id', 'value']);
    // The page never sees the mask's world.
    expect(await page.evaluate(() => Object.keys(window).length)).toBe(
      await page.evaluate(() => Object.keys(window).length),
    );
  });

  it('a page that overrides builtins in its own world does not affect masking (P7)', async () => {
    await load(FORM);
    await page.evaluate(() => {
      Document.prototype.querySelectorAll = () => {
        throw new Error('nope');
      };
      Set.prototype.has = () => false;
      Array.prototype.push = () => 0;
      CSSStyleDeclaration.prototype.getPropertyValue = () => 'transparent';
    });
    const session = maskSession();
    await session.apply();
    await session.settle();
    await session.verify();
    expect(
      await page.evaluate(
        () =>
          getComputedStyle(document.getElementById('ssn') as Element)
            .webkitTextFillColor,
      ),
    ).toBe(TRANSPARENT);
    await session.restore();
    expect(
      await page.evaluate(() =>
        document.getElementById('pin')?.getAttribute('style'),
      ),
    ).toBeNull();
  });

  it('masks by value: a "Reference" field and an unlabeled field_37 holding an SSN (P6)', async () => {
    await load(`<!doctype html><html lang="en"><body>
      <label for="ref">Reference</label> <input id="ref" value="900-00-1234">
      <input id="field_37" name="field_37" value="900 00 1234">
      <label for="echo">Note</label> <input id="echo" value="see 900-77-4321">
      <label for="city">City</label> <input id="city" value="Springfield">
    </body></html>`);
    const session = maskSession();
    await session.apply();
    expect([
      await fill('#ref'),
      await fill('#field_37'),
      await fill('#echo'),
      await fill('#city'),
    ]).toEqual([TRANSPARENT, TRANSPARENT, TRANSPARENT, 'rgb(0, 0, 0)']);
    await session.restore();
  });

  it('classifies by the accessible name: a 550-character label, a ::before label and an img alt label (P4, P4b)', async () => {
    const long = `${'Please read this carefully. '.repeat(19)}Social Security Number`;
    expect(long.length).toBeGreaterThan(550);
    await load(`<!doctype html><html lang="en"><head><style>
      .before::before { content: 'SSN'; }
    </style></head><body>
      <label for="long">${long}</label> <input id="long" value="x">
      <label for="css" class="before"></label> <input id="css" value="x">
      <label for="img"><img alt="Social Security Number" src="data:,"></label> <input id="img" value="x">
      <label for="city">City</label> <input id="city" value="x">
    </body></html>`);
    const session = maskSession();
    await session.apply();
    expect([
      await fill('#long'),
      await fill('#css'),
      await fill('#img'),
      await fill('#city'),
    ]).toEqual([TRANSPARENT, TRANSPARENT, TRANSPARENT, 'rgb(0, 0, 0)']);
    await session.restore();
  });

  it('masks the parent of a registered known value echoed as page text (I2)', async () => {
    await load(`<!doctype html><html lang="en"><body>
      <p id="confirm">You entered: <span id="echo">${KNOWN}</span></p>
      <p id="other">Nothing here</p>
    </body></html>`);
    const before = await styles();
    const session = maskSession();
    await session.apply();
    expect(await fill('#echo')).toBe(TRANSPARENT);
    expect(await fill('#other')).toBe('rgb(0, 0, 0)');
    await session.verify();
    await session.restore();
    expect(await styles()).toEqual(before);
  });

  it('hides ::selection, ::placeholder and descendants against page rules (C4)', async () => {
    await load(`<!doctype html><html lang="en"><head><style>
      ::selection { color: #000; background: #fff; -webkit-text-fill-color: #000; }
      ::placeholder { -webkit-text-fill-color: #000; }
      .v { -webkit-text-fill-color: #000 !important; }
      #ce .v { -webkit-text-fill-color: #000 !important; }
    </style></head><body>
      <label for="ssn">SSN</label> <input id="ssn" value="900-00-1234">
      <label for="sp">Spouse SSN</label> <input id="sp" placeholder="900-00-2468">
      <span id="cl">Card number</span>
      <div id="ce" contenteditable aria-labelledby="cl"><span class="v">4111</span></div>
    </body></html>`);
    const session = maskSession();
    await session.apply();
    expect({
      selection: await fill('#ssn', '::selection'),
      placeholder: await fill('#sp', '::placeholder'),
      child: await fill('#ce .v'),
      childSelection: await fill('#ce .v', '::selection'),
    }).toEqual({
      selection: TRANSPARENT,
      placeholder: TRANSPARENT,
      child: TRANSPARENT,
      childSelection: TRANSPARENT,
    });
    await session.verify();
    // A stronger page rule added during the window out-paints the mask.
    await page.addStyleTag({
      content: 'html body #ssn::selection { color: #000 !important; }',
    });
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
  });

  it('verify refuses when the page stripped a mask during the window', async () => {
    await load(FORM);
    const session = maskSession();
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

  it('verify refuses a React-style re-render that replaces the masked input (P1)', async () => {
    await load(FORM);
    const session = maskSession();
    await session.apply();
    await page.evaluate(() => {
      const old = document.getElementById('ssn') as HTMLInputElement;
      const fresh = document.createElement('input');
      fresh.id = 'ssn';
      fresh.value = old.value;
      old.replaceWith(fresh);
    });
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
  });

  it('verify refuses a clone swapped in for the capture and swapped back (P1b)', async () => {
    await load(FORM);
    const session = maskSession();
    await session.apply();
    await page.evaluate(() => {
      const old = document.getElementById('ssn') as HTMLInputElement;
      const clone = old.cloneNode(true) as HTMLInputElement;
      clone.removeAttribute('style');
      for (const a of clone.getAttributeNames()) {
        if (a.startsWith('data-labs-mask')) clone.removeAttribute(a);
      }
      old.replaceWith(clone);
      clone.replaceWith(old);
    });
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
  });

  it('verify refuses a srcdoc iframe with an SSN field appended during the window', async () => {
    await load(FORM);
    const session = maskSession();
    await session.apply();
    await page.evaluate(() => {
      const f = document.createElement('iframe');
      f.srcdoc = '<label for=s>SSN</label><input id=s value=900-00-1234>';
      document.body.append(f);
    });
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
  });

  it('verify refuses when an unmasked field takes a sensitive value during the window (C3)', async () => {
    await load(FORM);
    const session = maskSession();
    await session.apply();
    await session.verify();
    await page.fill('#city', '900-00-1234');
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore();
  });

  it('verify refuses after a navigation, and restore is then a no-op', async () => {
    await load(FORM);
    const session = maskSession();
    await session.apply();
    await load(FORM);
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    // The holder died with the old document: restore cannot reach it (and
    // says so), and the new document is left untouched.
    await session.restore().catch(() => {});
    expect(await fill('#ssn')).toBe('rgb(20, 20, 20)');
  });

  it('verify refuses when the document was rewritten in place (document.open)', async () => {
    await load(FORM);
    const session = maskSession();
    await session.apply();
    await page.setContent(FORM);
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await session.restore().catch(() => {});
  });

  it('refuses a closed shadow root holding a field (P3)', async () => {
    await load(`<!doctype html><html lang="en"><body><div id="host"></div><script>
      const root = document.getElementById('host').attachShadow({ mode: 'closed' });
      root.innerHTML = '<label>SSN <input value="900-00-1234"></label>';
    </script></body></html>`);
    const session = maskSession();
    await expect(session.apply()).rejects.toThrow();
    await session.restore();
    // An open one is walked and masked.
    await load(`<!doctype html><html lang="en"><body><div id="host"></div><script>
      const root = document.getElementById('host').attachShadow({ mode: 'open' });
      root.innerHTML = '<label>SSN <input id="in" value="900-00-1234"></label>';
    </script></body></html>`);
    const open = maskSession();
    await open.apply();
    expect(
      await page.evaluate(
        () =>
          getComputedStyle(
            document
              .getElementById('host')
              ?.shadowRoot?.getElementById('in') as Element,
          ).webkitTextFillColor,
      ),
    ).toBe(TRANSPARENT);
    await open.restore();
  });

  it('refuses when a session reports an uninspectable iframe', async () => {
    // A cross-site iframe is out of process: not in this session's frame
    // tree, so only a child session of its own could mask it.
    await page.route('http://main.test/', (r) =>
      r.fulfill({
        contentType: 'text/html',
        body: '<!doctype html><html lang="en"><body><iframe id="x" src="http://other.test/"></iframe></body></html>',
      }),
    );
    await page.route('http://other.test/', (r) =>
      r.fulfill({
        contentType: 'text/html',
        body: '<label for=s>SSN</label><input id=s value=900-00-1234>',
      }),
    );
    try {
      await page.goto('http://main.test/');
      await page.frameLocator('#x').locator('#s').waitFor();
      const tree = frameIdsOf(await send('Page.getFrameTree'));
      const consulted: string[] = [];
      const refused = maskSession({
        isCoveredFrame: (frameId) => {
          consulted.push(frameId);
          return false;
        },
      });
      await expect(refused.apply()).rejects.toThrow();
      await refused.restore();
      const oopif = consulted.filter((id) => !tree.includes(id));
      expect(oopif).toHaveLength(1);

      const covered = maskSession({
        isCoveredFrame: (frameId) => frameId === oopif[0],
      });
      await covered.apply();
      await covered.restore();
    } finally {
      await page.unrouteAll();
    }
  });

  it('an invisible frame owner paints nothing and is not a failure', async () => {
    await load(
      '<!doctype html><html lang="en"><body><iframe sandbox style="display:none" srcdoc="x"></iframe><iframe sandbox width="0" height="0" srcdoc="x"></iframe></body></html>',
    );
    const session = maskSession();
    await session.apply();
    await session.restore();
  });

  it('settle resolves after a rendered frame', async () => {
    await load(FORM);
    const session = maskSession({ outOfProcess: true });
    await session.apply();
    await session.settle();
    await session.restore();
  });
});

const frameIdsOf = (tree: unknown): string[] => {
  const out: string[] = [];
  const walk = (n: { frame: { id: string }; childFrames?: unknown[] }) => {
    out.push(n.frame.id);
    for (const c of n.childFrames ?? []) walk(c as typeof n);
  };
  walk((tree as { frameTree: Parameters<typeof walk>[0] }).frameTree);
  return out;
};

describe('createMaskSession with a stub send', () => {
  /**
   * A fake session: one frame, no fields, answering settle() with `how`
   * (or failing it). Records the methods sent.
   */
  const stubSession = (
    settle: Array<'raf' | 'timeout' | 'fail'>,
    outOfProcess: boolean,
  ) => {
    const methods: string[] = [];
    const stub = async (method: string, params?: Record<string, unknown>) => {
      const fn = String(params?.functionDeclaration ?? '');
      methods.push(fn ? fn.replace(/.*this\.(\w+).*/s, '$1') : method);
      if (method === 'Page.getFrameTree') {
        return { frameTree: { frame: { id: 'F' } } };
      }
      if (method === 'Page.createIsolatedWorld') {
        return { executionContextId: 1 };
      }
      if (method === 'Runtime.evaluate') return { result: { objectId: 'h' } };
      if (method === 'Accessibility.getFullAXTree') return { nodes: [] };
      if (method === 'Runtime.getProperties') return { result: [] };
      if (method === 'DOM.describeNode')
        return { node: { nodeName: '#document' } };
      if (fn.includes('report')) {
        return { result: { value: { fields: [], owners: [] } } };
      }
      if (fn.includes('fieldList') || fn.includes('doc()')) {
        return { result: { objectId: 'o' } };
      }
      if (fn.includes('mask(')) return { result: { value: 0 } };
      if (fn.includes('settle')) {
        const how = settle.shift();
        if (how === 'fail') throw new Error('gone');
        return { result: { value: how } };
      }
      return { result: { value: true } };
    };
    const session = createMaskSession({
      send: stub,
      objectGroup: 'g',
      isCoveredFrame: () => false,
      isSensitiveValue: () => false,
      hasKnownValue: () => false,
      outOfProcess,
    });
    return { session, methods };
  };

  it('settle reports its path: an out-of-process frame that only timed out is retried once, then refused (I5)', async () => {
    const twice = stubSession(['timeout', 'timeout'], true);
    await twice.session.apply();
    await expect(twice.session.settle()).rejects.toBeInstanceOf(
      ScreenshotMaskError,
    );
    expect(twice.methods.filter((m) => m === 'settle')).toHaveLength(2);

    const retried = stubSession(['timeout', 'raf'], true);
    await retried.session.apply();
    await expect(retried.session.settle()).resolves.toBeUndefined();

    // The tab's own renderer paints with the capture: a timeout is fine.
    const tab = stubSession(['timeout'], false);
    await tab.session.apply();
    await expect(tab.session.settle()).resolves.toBeUndefined();
  });

  it('a settle that fails refuses the screenshot (I4)', async () => {
    const failing = stubSession(['fail'], false);
    await failing.session.apply();
    await expect(failing.session.settle()).rejects.toBeInstanceOf(
      ScreenshotMaskError,
    );
  });

  it('rejects on a malformed reply instead of treating it as masked', async () => {
    const session = createMaskSession({
      send: async () => ({}),
      objectGroup: 'g',
      isCoveredFrame: () => true,
      isSensitiveValue: () => false,
      hasKnownValue: () => false,
    });
    await expect(session.apply()).rejects.toThrow();
  });

  it('restore without a successful apply sends only the group release; verify and settle refuse', async () => {
    const stub = vi.fn(
      async (_m: string, _p?: Record<string, unknown>) => ({}),
    );
    const session = createMaskSession({
      send: stub,
      objectGroup: 'g',
      isCoveredFrame: () => true,
      isSensitiveValue: () => false,
      hasKnownValue: () => false,
    });
    await session.restore();
    await expect(session.verify()).rejects.toBeInstanceOf(ScreenshotMaskError);
    await expect(session.settle()).rejects.toBeInstanceOf(ScreenshotMaskError);
    expect(stub.mock.calls.map((c) => c[0])).toEqual([
      'Runtime.releaseObjectGroup',
    ]);
  });

  it('an apply the bridge gave up on stops before masking once restore runs, and releases its group', async () => {
    const methods: string[] = [];
    let releaseReport = () => {};
    const stub = async (method: string, params?: Record<string, unknown>) => {
      const fn = String(params?.functionDeclaration ?? '');
      methods.push(fn ? fn.replace(/.*this\.(\w+).*/s, '$1') : method);
      if (method === 'Page.getFrameTree') {
        return { frameTree: { frame: { id: 'F' } } };
      }
      if (method === 'Page.createIsolatedWorld') {
        return { executionContextId: 1 };
      }
      if (method === 'Runtime.evaluate') return { result: { objectId: 'h' } };
      if (fn.includes('report')) {
        await new Promise<void>((resolve) => {
          releaseReport = resolve;
        });
        return {
          result: { value: { fields: [{ parts: {}, value: '' }], owners: [] } },
        };
      }
      return { result: { value: true } };
    };
    const session = createMaskSession({
      send: stub,
      objectGroup: 'g',
      isCoveredFrame: () => true,
      isSensitiveValue: () => false,
      hasKnownValue: () => false,
    });
    const applying = session.apply();
    await new Promise((r) => setTimeout(r, 10));
    await session.restore().catch(() => {});
    releaseReport();
    await expect(applying).rejects.toThrow();
    expect(methods).not.toContain('mask');
    expect(
      methods.filter((m) => m === 'Runtime.releaseObjectGroup').length,
    ).toBeGreaterThan(0);
  });
});
