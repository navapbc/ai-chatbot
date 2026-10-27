import { describe, expect, it } from 'vitest';
import {
  MAX_TAB_PATH,
  UNKNOWN_PAGE,
  describeTabUrl,
} from '../../extension/panel/lib/tab-url';

// What the allow-site card says about the tab's actual URL: origin plus path
// only. The path may carry an identifier, but the query and fragment (where
// most identifiers and tokens live) and any credentials never show.

describe('describeTabUrl', () => {
  it('shows the origin and the path', () => {
    expect(describeTabUrl('https://login.example.gov/sso/start')).toBe(
      'https://login.example.gov/sso/start',
    );
  });

  it('drops the query', () => {
    expect(
      describeTabUrl('https://login.example.gov/sso?case=Zelda&ssn=123'),
    ).toBe('https://login.example.gov/sso');
  });

  it('drops the fragment', () => {
    expect(describeTabUrl('https://login.example.gov/a#token=Zelda')).toBe(
      'https://login.example.gov/a',
    );
  });

  it('drops credentials', () => {
    const shown = describeTabUrl('https://zelda:hunter2@login.example.gov/a');
    expect(shown).toBe('https://login.example.gov/a');
    expect(shown).not.toMatch(/zelda|hunter2/);
  });

  it('keeps a port', () => {
    expect(describeTabUrl('https://localhost:8443/x')).toBe(
      'https://localhost:8443/x',
    );
  });

  it('cuts a long path at 80 characters with a marker', () => {
    const path = `/${'a'.repeat(200)}`;
    const shown = describeTabUrl(`https://example.gov${path}?q=1`);
    expect(shown).toBe(`https://example.gov${path.slice(0, MAX_TAB_PATH)}…`);
    expect(MAX_TAB_PATH).toBe(80);
  });

  it('does not cut a path of exactly 80 characters', () => {
    const path = `/${'b'.repeat(79)}`;
    expect(describeTabUrl(`https://example.gov${path}`)).toBe(
      `https://example.gov${path}`,
    );
  });

  it('cuts by code point, never inside a surrogate pair', () => {
    // A path the URL parser leaves as raw astral characters would be
    // percent-encoded; feed one through anyway and check no lone surrogate.
    const shown = describeTabUrl(
      `https://example.gov/${'\u{1F600}'.repeat(90)}`,
    );
    expect(shown.endsWith('…')).toBe(true);
    expect(shown).not.toMatch(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/);
  });

  it('lowercases an uppercase host', () => {
    expect(describeTabUrl('HTTPS://LOGIN.Example.GOV/Path')).toBe(
      'https://login.example.gov/Path',
    );
  });

  it('shows an IDN host in its punycode form', () => {
    expect(describeTabUrl('https://bücher.example/x')).toBe(
      'https://xn--bcher-kva.example/x',
    );
  });

  it('accepts http', () => {
    expect(describeTabUrl('http://127.0.0.1:3000/a?b=c')).toBe(
      'http://127.0.0.1:3000/a',
    );
  });

  it.each([
    ['undefined', undefined],
    ['an empty string', ''],
    ['a non-URL', 'not a url'],
    ['ftp:', 'ftp://files.example.gov/a'],
    ['chrome:', 'chrome://settings/passwords'],
    ['about:', 'about:blank'],
    ['data:', 'data:text/html,<b>x</b>'],
    ['javascript:', 'javascript:alert(1)'],
    ['file:', 'file:///Users/zelda/case.pdf'],
  ])('%s is an unknown page', (_name, url) => {
    expect(describeTabUrl(url)).toBe(UNKNOWN_PAGE);
    expect(UNKNOWN_PAGE).toBe('an unknown page');
  });

  it('a non-string is an unknown page', () => {
    expect(describeTabUrl(42 as unknown as string)).toBe(UNKNOWN_PAGE);
  });
});
