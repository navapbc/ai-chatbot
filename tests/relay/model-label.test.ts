import { describe, expect, it } from 'vitest';
import { modelLabel } from '../../extension/panel/lib/model-label';

describe('modelLabel', () => {
  it('names Claude and GPT ids', () => {
    expect(modelLabel('claude-opus-5')).toBe('Claude Opus 5');
    expect(modelLabel('claude-opus-4-7')).toBe('Claude Opus 4.7');
    expect(modelLabel('claude-haiku-4-5')).toBe('Claude Haiku 4.5');
    expect(modelLabel('gpt-5.4-mini')).toBe('GPT-5.4 Mini');
  });

  it('returns an unknown id unchanged', () => {
    expect(modelLabel('something-else')).toBe('something-else');
  });
});
