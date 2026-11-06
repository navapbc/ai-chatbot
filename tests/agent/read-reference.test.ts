import { describe, it, expect } from 'vitest';
import { readReferenceFile } from '@/agent/tools/read_reference';

describe('readReferenceFile', () => {
  it('reads an existing reference file', async () => {
    const result = await readReferenceFile('field-patterns.md');
    expect(result).toHaveProperty('content');
    if ('content' in result) {
      expect(result.content.length).toBeGreaterThan(0);
    }
  });

  it('strips a leading references/ prefix', async () => {
    const result = await readReferenceFile('references/field-patterns.md');
    expect(result).toHaveProperty('content');
  });

  it('denies path traversal outside the references dir', async () => {
    const result = await readReferenceFile('../../package.json');
    expect(result).toEqual({ error: 'Access denied: path must be within references' });
  });

  it('returns a not-found error for a missing file', async () => {
    const result = await readReferenceFile('does-not-exist.md');
    expect(result).toEqual({ error: 'File not found: does-not-exist.md' });
  });
});
