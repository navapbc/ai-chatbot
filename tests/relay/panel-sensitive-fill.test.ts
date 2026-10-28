import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { TYPE_SENSITIVE_COPY } from '../../extension/panel/lib/copy';
import {
  createSensitiveFill,
  type FillView,
} from '../../extension/panel/lib/sensitive-fill';
import type {
  TypeSensitiveCode,
  TypeSensitiveResult,
} from '../../extension/panel/lib/worker-api';
import { SensitiveFill } from '../../extension/panel/screens/sensitive-fill';

// The masked input's state machine and markup. The value used here is
// synthetic and never appears in a test name. The behaviors that need a DOM
// (typing into the input, unmount in React) run in the panel-shell e2e.

const ROOT = path.resolve(__dirname, '../..');
const SECRET = 'Zq9-Tangerine-Value-731';
const RUN = 'run-1';

type Deferred<T> = { promise: Promise<T>; resolve: (v: T) => void };
const deferred = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const CODES = Object.keys(TYPE_SENSITIVE_COPY) as TypeSensitiveCode[];

const setup = (
  reply: () => Promise<TypeSensitiveResult> = async () => ({ ok: true }),
) => {
  const typeSensitive = vi.fn(async (runId: string, value: string) => {
    void runId;
    void value;
    return reply();
  });
  const onFilled = vi.fn();
  const fill = createSensitiveFill({ runId: RUN, typeSensitive, onFilled });
  const views: FillView[] = [];
  fill.subscribe((v) => views.push(v));
  return { fill, typeSensitive, onFilled, views };
};

describe('createSensitiveFill', () => {
  it('types the trimmed value once, clears it, then reports filled', async () => {
    const { fill, typeSensitive, onFilled } = setup();
    fill.setValue(`  \n${SECRET}\t \n`);
    await fill.fill();
    expect(typeSensitive).toHaveBeenCalledTimes(1);
    expect(typeSensitive).toHaveBeenCalledWith(RUN, SECRET);
    expect(fill.view()).toEqual({ value: '', busy: false, code: null });
    expect(onFilled).toHaveBeenCalledTimes(1);
    // No argument at all: the callback cannot carry the value.
    expect(onFilled.mock.calls[0]).toEqual([]);
  });

  it('keeps inner whitespace', async () => {
    const { fill, typeSensitive } = setup();
    fill.setValue(' a  b ');
    await fill.fill();
    expect(typeSensitive).toHaveBeenCalledWith(RUN, 'a  b');
  });

  it('does nothing when the value is empty or only whitespace', async () => {
    const { fill, typeSensitive, onFilled } = setup();
    await fill.fill();
    fill.setValue(' \n\t ');
    await fill.fill();
    expect(typeSensitive).not.toHaveBeenCalled();
    expect(onFilled).not.toHaveBeenCalled();
  });

  it('calls the worker once for a second press while in flight', async () => {
    const gate = deferred<TypeSensitiveResult>();
    const { fill, typeSensitive, onFilled } = setup(() => gate.promise);
    fill.setValue(SECRET);
    const first = fill.fill();
    const second = fill.fill();
    expect(fill.view().busy).toBe(true);
    // Edits while in flight are ignored.
    fill.setValue('changed');
    expect(fill.view().value).toBe(SECRET);
    gate.resolve({ ok: true });
    await Promise.all([first, second]);
    expect(typeSensitive).toHaveBeenCalledTimes(1);
    expect(onFilled).toHaveBeenCalledTimes(1);
    expect(fill.view()).toEqual({ value: '', busy: false, code: null });
  });

  it.each(CODES)(
    'a refusal (%s) keeps the value and shows its code',
    async (code) => {
      const { fill, onFilled, views } = setup(async () => ({
        ok: false,
        code,
      }));
      fill.setValue(SECRET);
      await fill.fill();
      expect(onFilled).not.toHaveBeenCalled();
      expect(fill.view()).toEqual({ value: SECRET, busy: false, code });
      expect(TYPE_SENSITIVE_COPY[code]).toBeTruthy();
      expect(views.at(-1)?.code).toBe(code);
      // Retry after the caseworker focused the right field.
      fill.setValue(`${SECRET}!`);
      expect(fill.view().code).toBeNull();
    },
  );

  it('turns a throwing worker into worker_error without leaking the text', async () => {
    const { fill, onFilled } = setup(async () => {
      throw new Error(`boom ${SECRET}`);
    });
    fill.setValue(SECRET);
    await fill.fill();
    expect(onFilled).not.toHaveBeenCalled();
    expect(fill.view().code).toBe('worker_error');
    expect(JSON.stringify(fill.view().code)).not.toContain('boom');
  });

  it('allows a retry after a refusal and sends provided only after ok', async () => {
    let ok = false;
    const { fill, typeSensitive, onFilled } = setup(async () =>
      ok ? { ok: true } : { ok: false, code: 'no_focused_field' },
    );
    fill.setValue(SECRET);
    await fill.fill();
    expect(onFilled).not.toHaveBeenCalled();
    ok = true;
    await fill.fill();
    expect(typeSensitive).toHaveBeenCalledTimes(2);
    expect(onFilled).toHaveBeenCalledTimes(1);
  });

  it('dispose clears the value and ignores a result still in flight', async () => {
    const gate = deferred<TypeSensitiveResult>();
    const { fill, onFilled, views } = setup(() => gate.promise);
    fill.setValue(SECRET);
    const pending = fill.fill();
    const emitted = views.length;
    fill.dispose();
    expect(fill.view().value).toBe('');
    gate.resolve({ ok: true });
    await pending;
    expect(onFilled).not.toHaveBeenCalled();
    expect(views.length).toBe(emitted);
    fill.setValue(SECRET);
    expect(fill.view().value).toBe('');
  });

  it('dispose clears a kept value after a refusal', async () => {
    const { fill } = setup(async () => ({ ok: false, code: 'type_failed' }));
    fill.setValue(SECRET);
    await fill.fill();
    expect(fill.view().value).toBe(SECRET);
    fill.dispose();
    expect(fill.view().value).toBe('');
  });

  it('clear drops a typed value and emits it gone', () => {
    const { fill, views } = setup();
    fill.setValue(SECRET);
    fill.clear();
    expect(fill.view().value).toBe('');
    expect(JSON.stringify(views.at(-1))).not.toContain(SECRET);
  });

  it('never puts the value in an emitted view after success', async () => {
    const { fill, views } = setup();
    fill.setValue(SECRET);
    await fill.fill();
    const afterFill = views.slice(views.findIndex((v) => v.busy));
    // While busy the view still holds it (the input shows it); once settled
    // it is gone.
    expect(JSON.stringify(afterFill.at(-1))).not.toContain(SECRET);
  });
});

describe('SensitiveFill markup', () => {
  const html = (props: Record<string, unknown> = {}) =>
    renderToStaticMarkup(
      createElement(SensitiveFill, {
        runId: RUN,
        onFilled: () => {},
        onTypeMyself: () => {},
        ...props,
      }),
    );

  it('renders a masked, unnamed, un-remembered input', () => {
    const out = html();
    expect(out).toContain('type="password"');
    expect(out).toMatch(/autocomplete="off"/i);
    expect(out).toMatch(/spellcheck="false"/i);
    const input = out.match(/<input[^>]*>/)?.[0] ?? '';
    expect(input).not.toMatch(/\sname=/);
    expect(input).not.toMatch(/\sid=/);
  });

  it('shows the fixed instruction and labels', () => {
    const out = html();
    expect(out).toContain(
      'Click the field in the page, then press Fill it in.',
    );
    expect(out).toContain('aria-label="Show what I typed"');
    expect(out).toContain('aria-pressed="false"');
    expect(out).toContain('Fill it in');
    expect(out).toContain('I typed it in the page');
  });

  it('disables Fill it in while empty', () => {
    const out = html();
    const button = out.match(/<button[^>]*>Fill it in<\/button>/)?.[0] ?? '';
    expect(button).toContain('disabled');
  });

  it('disables every control when the card is locked', () => {
    const out = html({ disabled: true, typedMyself: true });
    const buttons = out.match(/<button[^>]*>/g) ?? [];
    expect(buttons).toHaveLength(3);
    for (const b of buttons) expect(b).toContain('disabled');
    expect(out).toMatch(/aria-pressed="true"[^>]*>(?:<[^>]*>)*I typed/);
  });
});

describe('where the value may live', () => {
  const read = (rel: string) => readFileSync(path.join(ROOT, rel), 'utf8');

  it('the Draft type and the answer path carry no value field', () => {
    const needsYou = read('extension/panel/screens/needs-you.tsx');
    const draft = needsYou.match(/type Draft =[\s\S]*?;\n/)?.[0] ?? '';
    expect(draft).toContain(
      "kind: 'sensitive'; choice: SensitiveChoice; filled?: boolean",
    );
    expect(needsYou).not.toMatch(/onFilled=\{[^}]*value/);
    // The sensitive answer is built from the request and a choice only.
    expect(needsYou).toContain('sensitiveAnswer(q.request, draft.choice)');
  });

  it('the reducer, the answer client and the run session never import the fill', () => {
    for (const rel of [
      'extension/panel/state.ts',
      'extension/panel/lib/answer-client.ts',
      'extension/panel/lib/run-session.ts',
    ]) {
      let src = '';
      try {
        src = read(rel);
      } catch {
        continue;
      }
      expect(src).not.toMatch(/sensitive-fill|typeSensitive/);
    }
  });

  it('the fill files never log or store', () => {
    for (const rel of [
      'extension/panel/lib/sensitive-fill.ts',
      'extension/panel/screens/sensitive-fill.tsx',
    ]) {
      const src = read(rel);
      expect(src).not.toMatch(
        /console\.|localStorage|sessionStorage|chrome\.storage|fetch\(|indexedDB/,
      );
    }
  });
});
