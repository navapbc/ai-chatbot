import { describe, expect, test } from 'vitest';
import { isSpecificToolAction } from '@/components/tool-icon';

describe('isSpecificToolAction (browser tool)', () => {
  test('fill with a value is specific', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'fill', value: '05/05/2026' })).toBe(true);
  });

  test('fill without a value is generic', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'fill' })).toBe(false);
  });

  test('type with text is specific', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'type', text: '432433425' })).toBe(true);
  });

  test('select with values is specific', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'select', values: ['Female'] })).toBe(true);
  });

  test('navigate with a url is specific', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'navigate', url: 'https://example.com' })).toBe(true);
  });

  test('navigate without a url is generic', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'navigate' })).toBe(false);
  });

  test('getbylabel with subaction fill is specific', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'getbylabel', label: 'Date of birth', subaction: 'fill' })).toBe(true);
  });

  test('getbylabel with subaction click is generic', () => {
    expect(isSpecificToolAction('tool-browser', { action: 'getbylabel', label: 'Submit', subaction: 'click' })).toBe(false);
  });

  test.each(['click', 'snapshot', 'screenshot', 'scroll', 'wait', 'hover', 'press', 'back', 'reload'])(
    '%s is generic',
    (action) => {
      expect(isSpecificToolAction('tool-browser', { action })).toBe(false);
    },
  );

  test('missing input is generic', () => {
    expect(isSpecificToolAction('tool-browser', undefined)).toBe(false);
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
