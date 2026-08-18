import { describe, expect, test } from 'vitest';
import { isSpecificToolAction } from '@/components/tool-icon';

describe('isSpecificToolAction (browser tool)', () => {
  test('fill with a value is specific', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['fill', '@e1', '05/05/2026'] })).toBe(true);
  });

  test('fill without a value is generic', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['fill', '@e1'] })).toBe(false);
  });

  test('type with text is specific', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['type', '@e1', '432433425'] })).toBe(true);
  });

  test('select with a value is specific', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['select', '@e1', 'Female'] })).toBe(true);
  });

  test('open with a url is specific', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['open', 'https://example.com'] })).toBe(true);
  });

  test('open without a url is generic', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['open'] })).toBe(false);
  });

  test('find label ... fill is specific', () => {
    expect(
      isSpecificToolAction('tool-browser', { command: ['find', 'label', 'Date of birth', 'fill', '01/01/2000'] }),
    ).toBe(true);
  });

  test('find label ... click is generic', () => {
    expect(isSpecificToolAction('tool-browser', { command: ['find', 'label', 'Submit', 'click'] })).toBe(false);
  });

  test('flags are not mistaken for positional values', () => {
    // `snapshot -s form` must not read "form" as a filled-in value.
    expect(isSpecificToolAction('tool-browser', { command: ['snapshot', '-s', 'form'] })).toBe(false);
  });

  test.each([
    ['click', ['click', '@e1']],
    ['snapshot', ['snapshot']],
    ['screenshot', ['screenshot']],
    ['scroll', ['scroll', 'down', '500']],
    ['wait', ['wait', '2000']],
    ['hover', ['hover', '@e1']],
    ['press', ['press', 'Enter']],
    ['back', ['back']],
    ['reload', ['reload']],
  ])('%s is generic', (_name, command) => {
    expect(isSpecificToolAction('tool-browser', { command })).toBe(false);
  });

  test('missing input is generic', () => {
    expect(isSpecificToolAction('tool-browser', undefined)).toBe(false);
  });

  test('empty command is generic', () => {
    expect(isSpecificToolAction('tool-browser', { command: [] })).toBe(false);
  });
});

describe('isSpecificToolAction (legacy browser_* / playwright_browser_* tools)', () => {
  test('browser_type with text is specific', () => {
    expect(isSpecificToolAction('tool-browser_type', { text: 'hello' })).toBe(true);
  });

  test('browser_type without text is generic', () => {
    expect(isSpecificToolAction('tool-browser_type', {})).toBe(false);
  });

  test('browser_select_option with values is specific', () => {
    expect(isSpecificToolAction('tool-browser_select_option', { values: ['Female'] })).toBe(true);
  });

  test('browser_navigate with url is specific', () => {
    expect(isSpecificToolAction('tool-playwright_browser_navigate', { url: 'https://example.com' })).toBe(true);
  });

  test('browser_click is generic', () => {
    expect(isSpecificToolAction('tool-browser_click', { element: 'button' })).toBe(false);
  });

  test('browser_snapshot is generic', () => {
    expect(isSpecificToolAction('tool-playwright_browser_snapshot', {})).toBe(false);
  });
});
