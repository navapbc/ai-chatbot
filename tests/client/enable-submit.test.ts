import { describe, expect, test, vi } from 'vitest';

vi.mock('@/lib/kernel/browser', () => ({
  getOrCreateBrowser: vi.fn(),
}));
vi.mock('agent-browser/dist/actions.js', () => ({
  executeCommand: vi.fn(),
}));

import { createEnableSubmitTool } from '@/lib/ai/tools/enable-submit';
import { phase0LocateButton } from '@/lib/ai/tools/enable-submit-phases';

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
@e1 [textbox name="firstName"] "John"
@e2 [button] "Submit Application"
`.trim();

  const SNAPSHOT_DISABLED = `
@e1 [textbox name="firstName"] "John"
@e2 [button disabled] "Submit Application"
`.trim();

  test('returns "enabled" short-circuit when button is not disabled', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_ENABLED });
    const result = await phase0LocateButton({ runCommand });
    expect(result.outcome).toEqual({ status: 'enabled' });
  });

  test('returns continue + selector when button is disabled', async () => {
    const runCommand = vi.fn().mockResolvedValue({ success: true, output: SNAPSHOT_DISABLED });
    const result = await phase0LocateButton({ runCommand });
    expect(result.outcome).toBeNull();
    expect(result.submitSelector).toBe('@e2');
  });

  test('respects an explicit submitSelector when passed', async () => {
    const runCommand = vi
      .fn()
      .mockResolvedValueOnce({ success: true, output: '@e1 [textbox name="firstName"] "John"' })
      .mockResolvedValueOnce({ success: true, output: 'false' });
    const result = await phase0LocateButton({ runCommand, submitSelector: '#mySubmit' });
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
