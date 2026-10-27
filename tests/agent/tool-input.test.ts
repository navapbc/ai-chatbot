import { describe, expect, it } from 'vitest';
import { redactToolInput } from '@/lib/relay/tool-input';

const H = '[hidden]';
const b = (argv: unknown, key = 'argv') => redactToolInput('browser', { [key]: argv });

describe('redactToolInput', () => {
  it('hides the value of fill, type and select', () => {
    expect(b(['fill', '@e3', '123-45-6789'])).toStrictEqual({ argv: ['fill', '@e3', H] });
    expect(b(['type', '@e2', 'abc'])).toStrictEqual({ argv: ['type', '@e2', H] });
    expect(b(['select', '@e4', 'Yes'])).toStrictEqual({ argv: ['select', '@e4', H] });
  });

  it('uses the real tool key, command, and keeps it', () => {
    expect(b(['fill', '@e3', 'secret'], 'command')).toStrictEqual({
      command: ['fill', '@e3', H],
    });
  });

  it('keeps a ref and a bare verb', () => {
    expect(b(['click', '@e1'])).toStrictEqual({ argv: ['click', '@e1'] });
    expect(b(['snapshot', '-i'])).toStrictEqual({ argv: ['snapshot', H] });
    expect(b(['snapshot'])).toStrictEqual({ argv: ['snapshot'] });
  });

  it('hides a non-ref second element', () => {
    expect(b(['press', 'Enter'])).toStrictEqual({ argv: ['press', H] });
    expect(b(['fill', '@e3x', 'v'])).toStrictEqual({ argv: ['fill', H, H] });
  });

  it('keeps only origin and path of an open url', () => {
    expect(b(['open', 'https://a.gov/x?ssn=1#f'])).toStrictEqual({
      argv: ['open', 'https://a.gov/x'],
    });
    const out = JSON.stringify(b(['open', 'https://user:pw@a.gov/x']));
    expect(out).not.toContain('user');
    expect(out).not.toContain('pw');
    expect(out).toContain('https://a.gov/x');
    expect(b(['open', 'not a url'])).toStrictEqual({ argv: ['open', H] });
  });

  it('hides everything after an open url too', () => {
    expect(b(['open', 'https://a.gov/x', 'extra'])).toStrictEqual({
      argv: ['open', 'https://a.gov/x', H],
    });
  });

  it('hides non-string elements', () => {
    expect(b(['fill', '@e1', { a: 1 }, 5, null])).toStrictEqual({
      argv: ['fill', '@e1', H, H, H],
    });
    expect(b([{ x: 1 }])).toStrictEqual({ argv: [H] });
  });

  it('returns {} for a bad shape', () => {
    expect(redactToolInput('browser', { argv: 'x' })).toStrictEqual({});
    expect(redactToolInput('browser', null)).toStrictEqual({});
    expect(redactToolInput('browser', 'fill')).toStrictEqual({});
    expect(redactToolInput('browser', {})).toStrictEqual({});
  });

  it('returns {} for ask_question, resolve_field_value and unknown tools', () => {
    const input = { prompt: 'SSN?', value: '123' };
    expect(redactToolInput('ask_question', input)).toStrictEqual({});
    expect(redactToolInput('resolve_field_value', input)).toStrictEqual({});
    expect(redactToolInput('whatever', input)).toStrictEqual({});
  });

  it('passes report_progress through', () => {
    const p = { sections: [], current: 'A' };
    expect(redactToolInput('report_progress', p)).toBe(p);
  });

  it('is safe against polluted input', () => {
    const evil = JSON.parse('{"__proto__":{"polluted":1},"argv":["fill","@e1","v"]}');
    expect(redactToolInput('browser', evil)).toStrictEqual({ argv: ['fill', '@e1', H] });
    expect(({} as any).polluted).toBeUndefined();
  });
});
