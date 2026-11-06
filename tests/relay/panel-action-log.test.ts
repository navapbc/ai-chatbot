import { describe, expect, it } from 'vitest';
import { createActionLog } from '../../extension/panel/lib/action-log';

describe('action log', () => {
  it('starts empty and returns a copy', () => {
    const log = createActionLog();
    expect(log.entries()).toEqual([]);
    log.add(['a']);
    log.entries().push('x');
    expect(log.entries()).toEqual(['a']);
  });

  it('the identical window twice adds nothing', () => {
    const log = createActionLog();
    log.add(['a', 'b', 'c', 'd', 'e']);
    log.add(['a', 'b', 'c', 'd', 'e']);
    expect(log.entries()).toEqual(['a', 'b', 'c', 'd', 'e']);
  });

  it('a sliding window of 5 adds only the new entries', () => {
    const log = createActionLog();
    log.add(['a', 'b', 'c', 'd', 'e']);
    log.add(['b', 'c', 'd', 'e', 'f']);
    log.add(['d', 'e', 'f', 'g', 'h']);
    expect(log.entries()).toEqual(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
  });

  it('a window with no overlap appends everything', () => {
    const log = createActionLog();
    log.add(['a', 'b']);
    log.add(['x', 'y', 'z']);
    expect(log.entries()).toEqual(['a', 'b', 'x', 'y', 'z']);
  });

  it('an overlap only in the middle of the log appends everything', () => {
    const log = createActionLog();
    log.add(['a', 'b', 'c', 'd']);
    log.add(['b', 'c', 'x']);
    expect(log.entries()).toEqual(['a', 'b', 'c', 'd', 'b', 'c', 'x']);
  });

  it('a shorter window that is a suffix of the log adds nothing', () => {
    const log = createActionLog();
    log.add(['a', 'b', 'c']);
    log.add(['b', 'c']);
    expect(log.entries()).toEqual(['a', 'b', 'c']);
  });

  it('trims, caps at 120 characters and drops empty entries', () => {
    const log = createActionLog();
    log.add(['  a  ', '', '   ', 'x'.repeat(200)]);
    expect(log.entries()).toEqual(['a', 'x'.repeat(120)]);
    log.add(['  a  ', '', 'x'.repeat(200)]);
    expect(log.entries()).toEqual(['a', 'x'.repeat(120)]);
  });

  it('caps at max and drops the oldest', () => {
    const log = createActionLog(3);
    log.add(['a', 'b']);
    log.add(['c', 'd', 'e']);
    expect(log.entries()).toEqual(['c', 'd', 'e']);
    const big = createActionLog();
    for (let i = 0; i < 250; i += 1) big.add([`n${i}`]);
    expect(big.entries()).toHaveLength(200);
    expect(big.entries()[0]).toBe('n50');
    expect(big.entries()[199]).toBe('n249');
  });

  it('handles a repeated action inside a window', () => {
    const log = createActionLog();
    log.add(['a']);
    log.add(['a', 'a', 'b']);
    expect(log.entries()).toEqual(['a', 'a', 'b']);
    log.add(['a', 'a', 'b']);
    expect(log.entries()).toEqual(['a', 'a', 'b']);
    log.add(['a', 'b', 'c']);
    expect(log.entries()).toEqual(['a', 'a', 'b', 'c']);
  });

  it('a window that goes backwards or repeats never throws', () => {
    const log = createActionLog();
    log.add(['a', 'b', 'c']);
    expect(() => log.add([])).not.toThrow();
    expect(() => log.add(['a'])).not.toThrow();
    expect(() => log.add(['a', 'b'])).not.toThrow();
    expect(log.entries().length).toBeGreaterThanOrEqual(3);
  });
});
