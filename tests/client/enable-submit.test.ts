import { describe, expect, test, vi } from 'vitest';

vi.mock('@/lib/kernel/browser', () => ({
  getOrCreateBrowser: vi.fn(),
}));
vi.mock('agent-browser/dist/actions.js', () => ({
  executeCommand: vi.fn(),
}));
import { createEnableSubmitTool } from '@/lib/ai/tools/enable-submit';
import { phase0LocateButton, phase1CheckRequiredFields, phase2ExpandSections, phase3WaitForTurnstile, phase4Verify, phase5Diagnose } from '@/lib/ai/tools/enable-submit-phases';

describe('enableSubmit tool', () => {
  test('factory returns a tool with a description and execute fn', () => {
    const tool = createEnableSubmitTool('chat-1', 'user-1');
    expect(tool).toBeDefined();
    expect(typeof tool.execute).toBe('function');
    expect(tool.description).toMatch(/submit/i);
  });
});

describe('phase0LocateButton', () => {
  const SNAPSHOT_ENABLED = `
- textbox "First Name" [ref=e1]
- button "Submit Application" [ref=e2]
`.trim();

  const SNAPSHOT_DISABLED = `
- textbox "First Name" [ref=e1]
- button "Submit Application" [ref=e2] [disabled]
`.trim();

  const SNAPSHOT_MULTIPLE_BUTTONS = `
- link "Apply now" [ref=e1]
- button "Send message" [ref=e2]
- button "Submit Application" [ref=e9] [disabled]
`.trim();

  test('returns "enabled" short-circuit when button is not disabled', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_ENABLED });
    const result = await phase0LocateButton({ runCommand });
    expect(result.outcome).toEqual({ status: 'enabled' });
    expect(result.submitSelector).toBe('@e2');
  });

  test('returns continue + selector when button is disabled', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_DISABLED });
    const result = await phase0LocateButton({ runCommand });
    expect(result.outcome).toBeNull();
    expect(result.submitSelector).toBe('@e2');
  });

  test('prefers the disabled button when multiple candidates exist', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_MULTIPLE_BUTTONS });
    const result = await phase0LocateButton({ runCommand });
    expect(result.submitSelector).toBe('@e9');
    expect(result.outcome).toBeNull();
  });

  test('ignores links even when label matches', async () => {
    const SNAPSHOT_LINK_ONLY = '- link "Apply now" [ref=e1]';
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_LINK_ONLY });
    const result = await phase0LocateButton({ runCommand });
    expect(result.outcome).toEqual({
      status: 'blocked-unknown',
      diagnostic: { reason: 'submit-button-not-found' },
    });
  });

  test('respects an explicit submitSelector when passed and button is enabled', async () => {
    const SNAPSHOT_NO_BUTTON_MATCH = '- textbox "First Name" [ref=e1]';
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_NO_BUTTON_MATCH });
    const result = await phase0LocateButton({ runCommand, submitSelector: '#mySubmit' });
    expect(result.outcome).toEqual({ status: 'enabled' });
    expect(result.submitSelector).toBe('#mySubmit');
  });
});

import { withSessionQueue } from '@/lib/ai/tools/browser';

describe('withSessionQueue export', () => {
  test('serializes calls per session id', async () => {
    const events: string[] = [];
    const a = withSessionQueue('s1', async () => {
      events.push('a-start');
      await new Promise((r) => setTimeout(r, 20));
      events.push('a-end');
      return 'a';
    });
    const b = withSessionQueue('s1', async () => {
      events.push('b-start');
      events.push('b-end');
      return 'b';
    });
    await Promise.all([a, b]);
    expect(events).toEqual(['a-start', 'a-end', 'b-start', 'b-end']);
  });
});

describe('phase1CheckRequiredFields', () => {
  const SNAPSHOT = '- textbox "First Name" [ref=e1]';

  const _model = {} as any;

  test('returns blocked-missing-fields when generateText reports missing', async () => {
    const _generateText = vi.fn().mockResolvedValue({ output: { missing: ['First Name', 'Last Name'] } });
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT });
    const result = await phase1CheckRequiredFields({ runCommand, _generateText: _generateText as any, _model });
    expect(result.outcome).toEqual({
      status: 'blocked-missing-fields',
      fields: ['First Name', 'Last Name'],
    });
  });

  test('returns null outcome when nothing is missing', async () => {
    const _generateText = vi.fn().mockResolvedValue({ output: { missing: [] } });
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT });
    const result = await phase1CheckRequiredFields({ runCommand, _generateText: _generateText as any, _model });
    expect(result.outcome).toBeNull();
  });

  test('falls back to "no missing" when generateText throws', async () => {
    const _generateText = vi.fn().mockRejectedValue(new Error('model timeout'));
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT });
    const result = await phase1CheckRequiredFields({ runCommand, _generateText: _generateText as any, _model });
    expect(result.outcome).toBeNull();
  });

  test('returns browser-error when snapshot fails', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: false, error: 'closed' });
    const result = await phase1CheckRequiredFields({ runCommand });
    expect(result.outcome).toEqual({ status: 'browser-error', error: 'closed' });
  });
});

describe('phase2ExpandSections', () => {
  const _model = {} as any;

  test('clicks each ref returned by the LLM, then re-snapshots', async () => {
    const _generateText = vi.fn().mockResolvedValue({ output: { refs: ['@e5', '@e7'] } });
    const runCommand = vi
      .fn()
      .mockResolvedValueOnce({ success: true, output: '- link "+ Expand" [ref=e5]' }) // initial snapshot
      .mockResolvedValueOnce({ success: true, output: 'ok' }) // click @e5
      .mockResolvedValueOnce({ success: true, output: 'ok' }) // click @e7
      .mockResolvedValueOnce({ success: true, output: 'fresh snapshot' }); // re-snapshot
    const result = await phase2ExpandSections({ runCommand, _generateText: _generateText as any, _model });
    expect(result.outcome).toBeNull();
    const calls = runCommand.mock.calls.map((c) => c[0]);
    expect(calls[1]).toEqual({ action: 'click', selector: '@e5' });
    expect(calls[2]).toEqual({ action: 'click', selector: '@e7' });
    expect(runCommand).toHaveBeenCalledTimes(4);
  });

  test('skips clicks and re-snapshot when LLM returns empty refs', async () => {
    const _generateText = vi.fn().mockResolvedValue({ output: { refs: [] } });
    const runCommand = vi.fn().mockResolvedValueOnce({ success: true, output: 'snap' });
    const result = await phase2ExpandSections({ runCommand, _generateText: _generateText as any, _model });
    expect(result.outcome).toBeNull();
    expect(runCommand).toHaveBeenCalledTimes(1); // initial snapshot only
  });

  test('falls back gracefully when generateText throws', async () => {
    const _generateText = vi.fn().mockRejectedValue(new Error('boom'));
    const runCommand = vi.fn().mockResolvedValueOnce({ success: true, output: 'snap' });
    const result = await phase2ExpandSections({ runCommand, _generateText: _generateText as any, _model });
    expect(result.outcome).toBeNull();
  });

  test('returns browser-error when initial snapshot fails', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: false, error: 'closed' });
    const result = await phase2ExpandSections({ runCommand });
    expect(result.outcome).toEqual({ status: 'browser-error', error: 'closed' });
  });
});

describe('phase3WaitForTurnstile', () => {
  test('returns enabled when disabled flips to false during polling', async () => {
    const runCommand = vi
      .fn()
      // tick 1: token empty, disabled=true
      .mockResolvedValueOnce({ success: true, output: '' })
      .mockResolvedValueOnce({ success: true, output: 'true' })
      // tick 2: token present, disabled=false -> exit
      .mockResolvedValueOnce({ success: true, output: 'TOKEN_VAL' })
      .mockResolvedValueOnce({ success: true, output: 'false' });
    const emit = vi.fn();
    const result = await phase3WaitForTurnstile(
      { runCommand, submitSelector: '@e2' },
      { tickMs: 1, maxTicks: 4, emit },
    );
    expect(result.outcome).toEqual({ status: 'enabled' });
    expect(emit).toHaveBeenCalled();
  });

  test('returns null after maxTicks when still disabled', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: 'true' });
    const emit = vi.fn();
    const result = await phase3WaitForTurnstile(
      { runCommand, submitSelector: '@e2' },
      { tickMs: 1, maxTicks: 2, emit },
    );
    expect(result.outcome).toBeNull();
    // emit called once per tick
    expect(emit.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  test('emit receives cumulative-seconds label for each tick', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: 'true' });
    const emit = vi.fn();
    await phase3WaitForTurnstile(
      { runCommand, submitSelector: '@e2' },
      { tickMs: 2000, maxTicks: 3, emit, _sleep: async () => {} },
    );
    // Match labels containing "(2s)", "(4s)", "(6s)" — cumulative seconds.
    const labels = emit.mock.calls.map((c) => c[0]);
    expect(labels.some((l) => l.includes('(2s)'))).toBe(true);
    expect(labels.some((l) => l.includes('(4s)'))).toBe(true);
    expect(labels.some((l) => l.includes('(6s)'))).toBe(true);
  });
});

describe('phase4Verify', () => {
  test('returns enabled when fresh snapshot shows button not disabled', async () => {
    const runCommand = vi
      .fn()
      .mockResolvedValueOnce({ success: true, output: '- button "Submit Application" [ref=e2]' });
    const result = await phase4Verify({ runCommand, submitSelector: '@e2' });
    expect(result.outcome).toEqual({ status: 'enabled' });
  });

  test('returns null when button still disabled', async () => {
    const runCommand = vi
      .fn()
      .mockResolvedValueOnce({ success: true, output: '- button "Submit Application" [ref=e2] [disabled]' });
    const result = await phase4Verify({ runCommand, submitSelector: '@e2' });
    expect(result.outcome).toBeNull();
  });

  test('returns browser-error when snapshot fails', async () => {
    const runCommand = vi.fn().mockResolvedValueOnce({ success: false, error: 'closed' });
    const result = await phase4Verify({ runCommand, submitSelector: '@e2' });
    expect(result.outcome).toEqual({ status: 'browser-error', error: 'closed' });
  });
});

describe('phase5Diagnose', () => {
  test('returns pending-turnstile when token empty', async () => {
    const runCommand = vi.fn().mockResolvedValueOnce({ success: true, output: '' });
    const result = await phase5Diagnose({ runCommand });
    expect(result.outcome).toEqual({
      status: 'pending-turnstile',
      message: 'Turnstile token is still empty — wait ~30s and try again.',
    });
    expect(result.tokenPresent).toBe(false);
  });

  test('returns null outcome + tokenPresent=true when token populated', async () => {
    const runCommand = vi.fn().mockResolvedValueOnce({ success: true, output: 'TOKEN_VALUE' });
    const result = await phase5Diagnose({ runCommand });
    expect(result.outcome).toBeNull();
    expect(result.tokenPresent).toBe(true);
  });

  test('returns pending-turnstile when token read fails', async () => {
    const runCommand = vi.fn().mockResolvedValueOnce({ success: false, error: 'eval blocked' });
    const result = await phase5Diagnose({ runCommand });
    // Treat browser-eval failure same as empty token — Phase 5's job is to recommend waiting, not surface evaluate failures.
    expect(result.outcome?.status).toBe('pending-turnstile');
    expect(result.tokenPresent).toBe(false);
  });
});
