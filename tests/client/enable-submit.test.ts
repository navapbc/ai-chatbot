import { describe, expect, test, vi } from 'vitest';

vi.mock('@/lib/kernel/browser', () => ({
  getOrCreateBrowser: vi.fn(),
}));
vi.mock('agent-browser/dist/actions.js', () => ({
  executeCommand: vi.fn(),
}));

import { createEnableSubmitTool } from '@/lib/ai/tools/enable-submit';

describe('enableSubmit tool', () => {
  test('factory returns a tool with a description and execute fn', () => {
    const tool = createEnableSubmitTool('chat-1', 'user-1');
    expect(tool).toBeDefined();
    expect(typeof tool.execute).toBe('function');
    expect(tool.description).toMatch(/submit/i);
  });
});
