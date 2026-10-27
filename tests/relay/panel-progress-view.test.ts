import { describe, expect, it } from 'vitest';
import { buildProgressView } from '../../extension/panel/lib/progress-view';
import type { RunProgress } from '../../lib/relay/progress';

type Section = RunProgress['sections'][number];
const sec = (over: Partial<Section> = {}): Section => ({
  name: 'Applicant',
  status: 'waiting',
  filled: 0,
  ...over,
});
const prog = (sections: Section[], current?: string): RunProgress => ({
  sections,
  ...(current ? { current } : {}),
});
const acts = (n: number) => Array.from({ length: n }, (_, i) => `a${i + 1}`);

describe('buildProgressView', () => {
  it('null progress: starting, no sections', () => {
    const v = buildProgressView(null, []);
    expect(v.bar).toEqual({ filled: 0, label: 'Starting...' });
    expect(v.sections).toEqual([]);
    expect(v.sectionLabel).toBeUndefined();
    expect(v.steps).toEqual([]);
  });

  it('null progress still lists steps from actions', () => {
    expect(buildProgressView(null, acts(3)).steps).toEqual(acts(3));
  });

  it('zero sections: steps are the last 8 actions', () => {
    const v = buildProgressView(prog([]), acts(12));
    expect(v.sections).toEqual([]);
    expect(v.steps).toEqual(acts(12).slice(4));
    expect(v.sectionLabel).toBeUndefined();
    expect(v.bar.label).toBe('0 fields so far');
  });

  it('keeps 40 sections and cuts 41 to 40', () => {
    const many = (n: number) =>
      Array.from({ length: n }, (_, i) => sec({ name: `S${i}` }));
    expect(buildProgressView(prog(many(40)), []).sections).toHaveLength(40);
    const v = buildProgressView(prog(many(41)), []);
    expect(v.sections).toHaveLength(40);
    expect(v.sectionLabel).toBe('Section 1 of 40');
  });

  it('cuts long names to 80, trims, defaults empty names', () => {
    const v = buildProgressView(
      prog([
        sec({ name: `  ${'x'.repeat(200)}  ` }),
        sec({ name: '   ' }),
        sec({ name: '' }),
      ]),
      [],
    );
    expect(v.sections[0].name).toBe('x'.repeat(80));
    expect(v.sections[1].name).toBe('Untitled section');
    expect(v.sections[2].name).toBe('Untitled section');
  });

  it('cuts action strings to 120', () => {
    const v = buildProgressView(prog([sec({ status: 'filling' })]), [
      'y'.repeat(300),
    ]);
    expect(v.sections[0].recent?.[0]).toBe('y'.repeat(120));
  });

  it('leaves markup as text', () => {
    const v = buildProgressView(
      prog([sec({ name: '<b>x</b>', status: 'filling' })]),
      ['<i>did</i>'],
    );
    expect(v.sections[0].name).toBe('<b>x</b>');
    expect(v.sections[0].recent).toEqual(['<i>did</i>']);
  });

  it('clamps counts', () => {
    const v = buildProgressView(
      prog([
        sec({ filled: -3, total: 5 }),
        sec({ filled: Number.NaN, total: 5 }),
        sec({ filled: 9, total: 5 }),
        sec({ filled: 2.9, total: -1 }),
      ]),
      [],
    );
    expect(v.sections.map((s) => s.filled)).toEqual([0, 0, 5, 0]);
    expect(v.sections[2].count).toBe('5 of 5');
    expect(v.sections[0].count).toBe('0 of 5');
    expect(v.sections[3].total).toBe(0);
  });

  it('total label only when every section has a total', () => {
    const all = buildProgressView(
      prog([
        sec({ filled: 2, total: 4 }),
        sec({ filled: 1, total: 6, status: 'filling' }),
      ]),
      [],
    );
    expect(all.bar).toEqual({ filled: 3, total: 10, label: '3 of 10 fields' });
    const mixed = buildProgressView(
      prog([sec({ filled: 2, total: 4 }), sec({ filled: 1 })]),
      [],
    );
    expect(mixed.bar).toEqual({ filled: 3, label: '3 fields so far' });
    expect(mixed.sections[1].count).toBe('1');
    expect(mixed.sections[0].count).toBe('2 of 4');
  });

  it('active section: current, else filling, needs_you, not done', () => {
    const base = [
      sec({ name: 'A', status: 'done' }),
      sec({ name: 'B', status: 'waiting' }),
      sec({ name: 'C', status: 'filling' }),
    ];
    expect(buildProgressView(prog(base, 'B'), []).sectionLabel).toBe(
      'Section 2 of 3',
    );
    // current names no section: falls back to the first filling.
    expect(buildProgressView(prog(base, 'Nope'), []).sectionLabel).toBe(
      'Section 3 of 3',
    );
    const needs = [
      sec({ name: 'A', status: 'done' }),
      sec({ name: 'B', status: 'needs_you' }),
      sec({ name: 'C', status: 'waiting' }),
    ];
    expect(buildProgressView(prog(needs), []).sectionLabel).toBe(
      'Section 2 of 3',
    );
    const waiting = [
      sec({ name: 'A', status: 'done' }),
      sec({ name: 'B', status: 'waiting' }),
    ];
    expect(buildProgressView(prog(waiting), []).sectionLabel).toBe(
      'Section 2 of 2',
    );
  });

  it('all done: X equals Y, nothing open', () => {
    const v = buildProgressView(
      prog([
        sec({ name: 'A', status: 'done', filled: 2 }),
        sec({ name: 'B', status: 'done', filled: 3 }),
      ]),
      ['a'],
    );
    expect(v.sectionLabel).toBe('Section 2 of 2');
    expect(v.sections.every((s) => !s.open)).toBe(true);
  });

  it('one filling section opens and carries the last 5 actions', () => {
    const v = buildProgressView(
      prog([
        sec({ name: 'A', status: 'done', filled: 4, total: 4 }),
        sec({ name: 'B', status: 'filling', filled: 1, total: 3 }),
        sec({ name: 'C', status: 'waiting' }),
      ]),
      acts(7),
    );
    expect(v.sectionLabel).toBe('Section 2 of 3');
    expect(v.sections.map((s) => s.open)).toEqual([false, true, false]);
    expect(v.sections[1].recent).toEqual(acts(7).slice(2));
    expect(v.sections[0].recent).toBeUndefined();
    expect(v.sections[2].recent).toBeUndefined();
    expect(v.sections[2].status).toBe('waiting');
    expect(v.steps).toEqual([]);
  });

  it('needs_you is kept and, when active, open', () => {
    const v = buildProgressView(
      prog([sec({ name: 'A', status: 'needs_you' })]),
      [],
    );
    expect(v.sections[0].status).toBe('needs_you');
    expect(v.sections[0].open).toBe(true);
  });

  it('a done section named by current stays collapsed', () => {
    const v = buildProgressView(
      prog([sec({ name: 'A', status: 'done' }), sec({ name: 'B' })], 'A'),
      [],
    );
    expect(v.sections[0].open).toBe(false);
  });

  it('never throws on garbage', () => {
    const bad = [
      undefined,
      0,
      'x',
      [],
      { sections: 'x', actions: 5 },
      { sections: [null, 1, 'a', {}, { name: 5, status: 'zzz', filled: 'q' }] },
      { sections: null },
    ];
    for (const p of bad) {
      for (const a of [undefined, null, 'x', 5, [1, null, 'ok'], {}]) {
        expect(() => buildProgressView(p as never, a as never)).not.toThrow();
      }
    }
    const v = buildProgressView(
      { sections: [{ name: 5, status: 'zzz', filled: 'q' }] } as never,
      [1, null, 'ok'] as never,
    );
    expect(v.sections[0].name).toBe('Untitled section');
    expect(v.sections[0].status).toBe('waiting');
    expect(v.sections[0].filled).toBe(0);
  });
});
