import { describe, expect, test, vi } from 'vitest';

vi.mock('@/lib/kernel/browser', () => ({
  getOrCreateBrowser: vi.fn(),
}));
vi.mock('agent-browser/dist/actions.js', () => ({
  executeCommand: vi.fn(),
}));
import { createEnableSubmitTool } from '@/lib/ai/tools/enable-submit';
import { phase0LocateButton, phase1CheckRequiredFields } from '@/lib/ai/tools/enable-submit-phases';

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

  test('returns blocked-missing-fields when generateObject reports missing', async () => {
    const _generateObject = vi.fn().mockResolvedValue({ object: { missing: ['First Name', 'Last Name'] } });
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT });
    const result = await phase1CheckRequiredFields({ runCommand, _generateObject: _generateObject as any, _model });
    expect(result.outcome).toEqual({
      status: 'blocked-missing-fields',
      fields: ['First Name', 'Last Name'],
    });
  });

  test('returns null outcome when nothing is missing', async () => {
    const _generateObject = vi.fn().mockResolvedValue({ object: { missing: [] } });
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT });
    const result = await phase1CheckRequiredFields({ runCommand, _generateObject: _generateObject as any, _model });
    expect(result.outcome).toBeNull();
  });

  test('falls back to "no missing" when generateObject throws', async () => {
    const _generateObject = vi.fn().mockRejectedValue(new Error('model timeout'));
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT });
    const result = await phase1CheckRequiredFields({ runCommand, _generateObject: _generateObject as any, _model });
    expect(result.outcome).toBeNull();
  });

  test('returns browser-error when snapshot fails', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: false, error: 'closed' });
    const result = await phase1CheckRequiredFields({ runCommand });
    expect(result.outcome).toEqual({ status: 'browser-error', error: 'closed' });
  });
});
