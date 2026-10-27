import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  extensionOrigin,
  originRefusal,
  resetOriginWarningForTests,
} from '@/lib/relay/request-origin';

const ID = 'abcdefghijklmnopabcdefghijklmnop';
const ORIGIN = `chrome-extension://${ID}`;
const env = (value?: string) => ({ RELAY_EXTENSION_ORIGIN: value });
const req = (origin?: string) =>
  new Request('http://localhost/x', {
    headers: origin === undefined ? {} : { Origin: origin },
  });

let warn: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  resetOriginWarningForTests();
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
});

describe('extensionOrigin', () => {
  it('is null when unset, without a warning', () => {
    expect(extensionOrigin({})).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });

  it('returns a valid origin as given', () => {
    expect(extensionOrigin(env(ORIGIN))).toBe(ORIGIN);
    expect(warn).not.toHaveBeenCalled();
  });

  it.each([
    '',
    ID,
    `${ORIGIN}/`,
    ORIGIN.toUpperCase(),
    `chrome-extension://${ID.slice(1)}`,
    `chrome-extension://${ID}a`,
    `chrome-extension://${'q'.repeat(32)}`,
    ` ${ORIGIN}`,
    `https://${ID}`,
  ])('treats %j as unset', (value) => {
    expect(extensionOrigin(env(value))).toBeNull();
    expect(
      originRefusal(req('https://evil.example'), {}, env(value)),
    ).toBeNull();
  });

  it('logs a malformed value once per process, without the value', () => {
    const bad = 'chrome-extension://SECRETISH/';
    extensionOrigin(env(bad));
    extensionOrigin(env(bad));
    extensionOrigin(env(ID));
    expect(warn).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(warn.mock.calls)).not.toContain('SECRETISH');
    expect(warn.mock.calls[0]).toEqual([
      '[relay-origin] ignoring malformed RELAY_EXTENSION_ORIGIN',
    ]);
  });
});

describe('originRefusal', () => {
  it('never refuses with the env unset', () => {
    expect(
      originRefusal(req('https://x'), { requireOrigin: true }, {}),
    ).toBeNull();
    expect(originRefusal(req(), { requireOrigin: true }, {})).toBeNull();
  });

  it('accepts the exact origin', () => {
    expect(
      originRefusal(req(ORIGIN), { requireOrigin: true }, env(ORIGIN)),
    ).toBeNull();
  });

  it('refuses a different origin on any method setting', () => {
    expect(originRefusal(req('https://evil.example'), {}, env(ORIGIN))).toBe(
      'mismatch',
    );
    expect(
      originRefusal(
        req('https://evil.example'),
        { requireOrigin: true },
        env(ORIGIN),
      ),
    ).toBe('mismatch');
  });

  it('refuses a missing origin only when required', () => {
    expect(originRefusal(req(), {}, env(ORIGIN))).toBeNull();
    expect(
      originRefusal(req(), { requireOrigin: false }, env(ORIGIN)),
    ).toBeNull();
    expect(originRefusal(req(), { requireOrigin: true }, env(ORIGIN))).toBe(
      'missing',
    );
  });

  it('matches case-sensitively', () => {
    const upper = `CHROME-EXTENSION://${ID}`;
    const host = `chrome-extension://${ID.toUpperCase()}`;
    expect(originRefusal(req(upper), {}, env(ORIGIN))).toBe('mismatch');
    expect(originRefusal(req(host), {}, env(ORIGIN))).toBe('mismatch');
    expect(originRefusal(req(`${ORIGIN}/`), {}, env(ORIGIN))).toBe('mismatch');
  });
});
