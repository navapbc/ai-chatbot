import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  CONFIRMED_COPY,
  SUBMIT_COPY,
  unfilledStatusText,
} from '../../extension/panel/lib/copy';
import type { SubmissionView } from '../../extension/panel/lib/run-session';
import { createUnfilledField } from '../../extension/panel/lib/unfilled-field';
import type { UnfilledFieldResult } from '../../extension/panel/lib/worker-api';
import {
  ConfirmedScreen,
  STOP_FAILED_TEXT,
} from '../../extension/panel/screens/confirmed';
import { ReviewPanel } from '../../extension/panel/screens/review';
import {
  buildReviewModel,
  isConfirming,
  reviewActions,
  reviewLogNote,
} from '../../extension/panel/lib/review-model';

const sec = (
  name: string,
  status: 'waiting' | 'filling' | 'done' | 'needs_you',
  filled: number,
  total?: number,
) => ({ name, status, filled, ...(total === undefined ? {} : { total }) });

describe('buildReviewModel', () => {
  it('lists needs_you sections', () => {
    const m = buildReviewModel({
      sections: [sec('Income', 'needs_you', 2, 5), sec('Name', 'done', 3, 3)],
    });
    expect(m.hasProgress).toBe(true);
    expect(m.left).toEqual([
      { name: 'Income', reason: 'needs_you', count: '2 of 5' },
    ]);
  });

  it('lists a section whose filled is below a known total', () => {
    const m = buildReviewModel({ sections: [sec('Address', 'done', 1, 4)] });
    expect(m.left).toEqual([
      { name: 'Address', reason: 'incomplete', count: '1 of 4' },
    ]);
  });

  it('omits a section filled to its total, or with no total', () => {
    const m = buildReviewModel({
      sections: [sec('A', 'done', 4, 4), sec('B', 'filling', 2)],
    });
    expect(m.left).toEqual([]);
    expect(m.hasProgress).toBe(true);
  });

  it('has no progress for null or no sections', () => {
    expect(buildReviewModel(null)).toEqual({ left: [], hasProgress: false });
    expect(buildReviewModel({ sections: [] })).toEqual({
      left: [],
      hasProgress: false,
    });
    expect(buildReviewModel('x' as never).hasProgress).toBe(false);
  });

  it('caps at 40 sections', () => {
    const sections = Array.from({ length: 60 }, (_, i) =>
      sec(`S${i}`, 'needs_you', 0, 1),
    );
    expect(buildReviewModel({ sections }).left).toHaveLength(40);
  });

  it('sanitizes hostile names and counts', () => {
    const m = buildReviewModel({
      sections: [
        {
          name: `  ${'x'.repeat(200)}`,
          status: 'needs_you',
          filled: Number.POSITIVE_INFINITY,
          total: 1e12,
        },
        { name: 42, status: 'bogus', filled: -3, total: 5 } as never,
        null as never,
      ],
    });
    expect(Array.from(m.left[0].name)).toHaveLength(80);
    expect(m.left[0].count).toBe('0 of 1000000');
    expect(m.left[1]).toEqual({
      name: 'Untitled section',
      reason: 'incomplete',
      count: '0 of 5',
    });
  });

  it('never throws on junk', () => {
    expect(() => buildReviewModel({ sections: 7 } as never)).not.toThrow();
    expect(buildReviewModel({ sections: 7 } as never).hasProgress).toBe(false);
  });
});

describe('reviewActions', () => {
  it('keeps the last 50 strings, cut to 120 code points', () => {
    const log = Array.from({ length: 80 }, (_, i) => `a${i}`);
    const out = reviewActions([...log, 5 as never, 'y'.repeat(300)]);
    expect(out).toHaveLength(50);
    expect(Array.from(out[49])).toHaveLength(120);
    expect(out[0]).toBe('a31');
  });
  it('is empty for junk', () => {
    expect(reviewActions(undefined as never)).toEqual([]);
  });
});

describe('isConfirming', () => {
  it('is true only for a run ended through "I submitted it"', () => {
    expect(isConfirming({ endedReason: 'submitted' })).toBe(true);
    expect(isConfirming({ endedReason: 'user' })).toBe(false);
    expect(isConfirming({})).toBe(false);
    expect(isConfirming(null)).toBe(false);
  });
});

describe('ConfirmedScreen', () => {
  const html = (
    stopFailed: boolean,
    extra: Partial<Parameters<typeof ConfirmedScreen>[0]> = {},
  ) =>
    renderToStaticMarkup(
      createElement(ConfirmedScreen, {
        siteName: 'example.gov',
        stopFailed,
        onStartAnother: () => {},
        onHome: () => {},
        ...extra,
      }),
    );
  it('detected with a confirmation number: says the form was submitted and shows the number as text', () => {
    const out = html(false, {
      how: 'detected',
      confirmation: { token: 'AB-1234', kind: 'confirmation' },
    });
    expect(out).toContain('The form was submitted on example.gov.');
    expect(out).toContain('The page moved on after the form was sent.');
    expect(out).toContain('Confirmation number');
    expect(out).toContain('>AB-1234<');
    expect(out).not.toContain('You said');
  });
  it('a number after another label is not called a confirmation number', () => {
    const out = html(false, {
      how: 'detected',
      confirmation: { token: '20240017', kind: 'other' },
    });
    expect(out).toContain('Number shown on the page');
    expect(out).not.toContain('Confirmation number');
    expect(out).toContain('>20240017<');
  });
  it('detected with no number shows no number line; manual says what the caseworker said', () => {
    const detected = html(false, { how: 'detected' });
    expect(detected).toContain('The form was submitted on example.gov.');
    expect(detected).not.toMatch(/Confirmation number|Number shown/);
    const manual = html(false, { how: 'manual' });
    expect(manual).toContain('You said you submitted the form on example.gov.');
    expect(manual).not.toContain('The form was submitted');
  });
  it('escapes a hostile token (React text only)', () => {
    const out = html(false, {
      how: 'detected',
      confirmation: { token: '<img src=x>', kind: 'confirmation' },
    });
    expect(out).not.toContain('<img');
  });
  it('says what the caseworker said, with no stop notice when the stop worked', () => {
    const out = html(false);
    expect(out).toContain('You said you submitted the form on example.gov.');
    expect(out).not.toContain('could not reach');
    expect(out).not.toContain('>Stop<');
  });
  it('shows the fixed notice, and no Stop button, when the stop failed', () => {
    const out = html(true);
    expect(out).toContain(STOP_FAILED_TEXT.replace(/'/g, '&#x27;'));
    expect(out).not.toContain('>Stop<');
  });
});

describe('reviewLogNote', () => {
  it('notes a mid-run open as fixed text, only then', () => {
    expect(reviewLogNote(false)).toBeNull();
    expect(reviewLogNote(true)).toBe(
      'The panel was opened while the run was under way, so steps before then are not shown.',
    );
  });
});

describe('Review: Show me the unfilled field', () => {
  const html = renderToStaticMarkup(
    createElement(ReviewPanel, {
      runId: 'run-1',
      progress: null,
      actions: [],
      stopping: false,
      onSubmitted: () => {},
      onStop: () => {},
    }),
  );

  it('renders the button above I submitted it, enabled, with an empty status line', () => {
    const show = html.indexOf('Show me the unfilled field');
    const submitted = html.indexOf('I submitted it');
    expect(show).toBeGreaterThan(-1);
    expect(show).toBeLessThan(submitted);
    expect(html).toMatch(/role="status"[^>]*aria-live="polite"[^>]*><\/p>/);
    expect(html).toMatch(/aria-disabled="false"/);
  });

  const text = (r: UnfilledFieldResult) => unfilledStatusText(r);

  it('found: count and the look-for sentence, no promise of a box', () => {
    const t = text({ ok: true, found: true, remaining: 3 });
    expect(t).toContain('Showing the next field that is still empty. 3 left.');
    expect(t).toContain('highlighted or focused');
    expect(t).not.toMatch(/sensitive|many/);
    expect(text({ ok: true, found: true, remaining: 1 })).toContain('1 left.');
  });

  it('found and sensitive, and capped, add their sentences', () => {
    const t = text({
      ok: true,
      found: true,
      remaining: 12,
      sensitive: true,
      capped: true,
    });
    expect(t).toContain('sensitive field');
    expect(t).toContain('fill it in yourself in the page');
    expect(t).not.toContain('Fill it in');
    expect(t).toContain('There are many; showing the first.');
  });

  it('not found', () => {
    expect(text({ ok: true, found: false, remaining: 0 })).toBe(
      'No empty required fields found. Check the page before you submit.',
    );
  });

  it.each([
    ['run_held', /paused while the tab is on another site/],
    ['not_attached', /not connected to this tab/],
    ['registration_pending', /Wait a moment and try again/],
    ['scan_failed', /Could not check the page/],
    ['worker_error', /Could not check the page/],
    ['field_out_of_scope', /outside the page/],
  ] as const)('refusal %s has its own sentence', (code, re) => {
    expect(text({ ok: false, code })).toMatch(re);
  });

  it('a double press makes one call with the run id', async () => {
    let release!: (r: UnfilledFieldResult) => void;
    const show = vi
      .fn(async (runId: string): Promise<UnfilledFieldResult> => {
        void runId;
        return { ok: true, found: false, remaining: 0 };
      })
      .mockImplementationOnce(
        (runId: string) =>
          new Promise<UnfilledFieldResult>((resolve) => {
            void runId;
            release = resolve;
          }),
      );
    const c = createUnfilledField({ runId: 'run-7', showUnfilledField: show });
    const first = c.press();
    const second = c.press();
    expect(c.view().busy).toBe(true);
    release({ ok: true, found: false, remaining: 0 });
    await Promise.all([first, second]);
    expect(show).toHaveBeenCalledTimes(1);
    expect(show).toHaveBeenCalledWith('run-7');
    expect(c.view()).toEqual({
      busy: false,
      result: { ok: true, found: false, remaining: 0 },
    });
    await c.press();
    expect(show).toHaveBeenCalledTimes(2);
  });

  it('a throwing call ends as worker_error and is not busy; dispose silences', async () => {
    const c = createUnfilledField({
      runId: 'r',
      showUnfilledField: async () => {
        throw new Error('x');
      },
    });
    await c.press();
    expect(c.view()).toEqual({
      busy: false,
      result: { ok: false, code: 'worker_error' },
    });
    const seen = vi.fn();
    c.subscribe(seen);
    const p = c.press();
    c.dispose();
    await p;
    expect(seen).toHaveBeenCalledTimes(1);
  });
});

describe('Review: a detected submission', () => {
  const render = (submission?: SubmissionView, stopping = false) =>
    renderToStaticMarkup(
      createElement(ReviewPanel, {
        runId: 'run-1',
        progress: null,
        actions: [],
        stopping,
        submission,
        onSubmitted: () => {},
        onStop: () => {},
        onConfirmSubmission: () => {},
        onDismissSubmission: () => {},
      }),
    );

  it('no longer says submitting is only by hand, and keeps I submitted it', () => {
    const out = render();
    expect(out).toContain(SUBMIT_COPY.reviewHint);
    expect(out).toContain('I submitted it');
    expect(out).not.toContain(SUBMIT_COPY.checking);
    expect(out).not.toContain(SUBMIT_COPY.askTitle);
    // Only the unfilled-field status line: no empty extra one.
    expect(out.match(/role="status"/g)).toHaveLength(1);
  });

  it('checking: a status line, no prompt', () => {
    const out = render({ stage: 'checking' });
    expect(out).toContain(SUBMIT_COPY.checking);
    expect(out).not.toContain(SUBMIT_COPY.askTitle);
    expect(out.match(/role="status"/g)).toHaveLength(2);
  });

  it('ask: the question with Yes and No, I submitted it still there', () => {
    const out = render({ stage: 'ask' });
    expect(out).toContain(SUBMIT_COPY.askTitle);
    expect(out).toContain(SUBMIT_COPY.askBody);
    expect(out).toContain(SUBMIT_COPY.yes);
    expect(out).toContain(SUBMIT_COPY.no);
    expect(out).toContain('I submitted it');
    expect(out).not.toContain(SUBMIT_COPY.numberOnPage);
    expect(out).not.toContain(SUBMIT_COPY.checking);
  });

  it('ask with a number after another label shows it as text', () => {
    const out = render({
      stage: 'ask',
      confirmation: { token: '20240017', kind: 'other' },
    });
    expect(out).toContain(SUBMIT_COPY.numberOnPage);
    expect(out).toContain('>20240017<');
    const hostile = render({
      stage: 'ask',
      confirmation: { token: '<b>x</b>', kind: 'other' },
    });
    expect(hostile).not.toContain('<b>x');
  });

  it('copy never claims the form is complete or accepted', () => {
    const all = [
      ...Object.values(SUBMIT_COPY),
      CONFIRMED_COPY.detected('x'),
      CONFIRMED_COPY.detectedDetail,
    ].join(' ');
    expect(all).not.toMatch(/complete|accepted|approved|success/i);
  });
});
