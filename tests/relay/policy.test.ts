import { describe, expect, it } from 'vitest';
import {
  checkCommand,
  inScope,
  isMethodAllowed,
  normalizeOrigin,
} from '@/lib/relay/policy';

const ctx = { origin: 'https://a.gov' };

describe('normalizeOrigin', () => {
  it('returns the origin for https URLs', () => {
    expect(normalizeOrigin('https://a.gov/x?y')).toBe('https://a.gov');
    expect(normalizeOrigin('https://a.gov:8443/x')).toBe('https://a.gov:8443');
  });

  it('returns null for non-https and unparseable input', () => {
    for (const url of [
      'http://a.gov',
      'about:blank',
      'chrome://x',
      'not a url',
      '',
    ]) {
      expect(normalizeOrigin(url)).toBeNull();
    }
  });
});

describe('inScope', () => {
  it('matches the exact origin', () => {
    expect(inScope('https://a.gov/page', 'https://a.gov')).toBe(true);
  });

  it('rejects a different port, subdomain, lookalike, scheme or junk', () => {
    expect(inScope('https://a.gov:8443/', 'https://a.gov')).toBe(false);
    expect(inScope('https://sub.a.gov/', 'https://a.gov')).toBe(false);
    expect(inScope('https://a.gov.evil.com/', 'https://a.gov')).toBe(false);
    expect(inScope('https://evil.com/?https://a.gov', 'https://a.gov')).toBe(
      false,
    );
    expect(inScope('http://a.gov/', 'https://a.gov')).toBe(false);
    expect(inScope('about:blank', 'https://a.gov')).toBe(false);
    expect(inScope('', 'https://a.gov')).toBe(false);
  });
});

describe('isMethodAllowed', () => {
  it('allows listed methods and refuses unknown ones', () => {
    expect(isMethodAllowed('Runtime.evaluate')).toBe(true);
    expect(isMethodAllowed('Network.getCookies')).toBe(false);
    expect(isMethodAllowed('')).toBe(false);
  });
});

describe('checkCommand', () => {
  it('refuses cookie, storage, fetch and download methods', () => {
    for (const method of [
      'Network.getAllCookies',
      'Network.getCookies',
      'Storage.getCookies',
      'Fetch.enable',
      'Page.setDownloadBehavior',
      'Not.AMethod',
    ]) {
      expect(checkCommand({ method }, ctx).ok).toBe(false);
    }
  });

  it('allows ordinary page-driving methods', () => {
    expect(
      checkCommand({ method: 'Accessibility.getFullAXTree' }, ctx),
    ).toEqual({ ok: true });
    expect(
      checkCommand({ method: 'Input.insertText', params: { text: 'x' } }, ctx),
    ).toEqual({ ok: true });
  });

  it('checks Page.navigate against scope', () => {
    expect(
      checkCommand(
        { method: 'Page.navigate', params: { url: 'https://a.gov/next' } },
        ctx,
      ),
    ).toEqual({ ok: true });
    const out = checkCommand(
      { method: 'Page.navigate', params: { url: 'https://b.gov/' } },
      ctx,
    );
    expect(out.ok).toBe(false);
  });

  it('checks Target.createTarget against scope, including a missing url', () => {
    expect(
      checkCommand(
        { method: 'Target.createTarget', params: { url: 'https://a.gov/' } },
        ctx,
      ),
    ).toEqual({ ok: true });
    expect(
      checkCommand(
        { method: 'Target.createTarget', params: { url: 'https://b.gov/' } },
        ctx,
      ).ok,
    ).toBe(false);
    expect(checkCommand({ method: 'Target.createTarget' }, ctx).ok).toBe(false);
    expect(
      checkCommand({ method: 'Target.createTarget', params: {} }, ctx).ok,
    ).toBe(false);
  });

  it('refuses malformed params without throwing', () => {
    for (const params of [undefined, null, 'x', 5, [], {}, { url: 5 }]) {
      expect(checkCommand({ method: 'Page.navigate', params }, ctx).ok).toBe(
        false,
      );
    }
  });
});
