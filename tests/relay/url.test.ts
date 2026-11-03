import { describe, expect, it } from 'vitest';
import { LOOPBACK_HOSTS, toRelayWsUrl } from '@/lib/relay/url';

describe('toRelayWsUrl', () => {
  it.each([
    ['https://user:pw@relay.example.com', null],
    ['https://relay.example.com?x=1', null],
    ['https://relay.example.com#f', null],
    ['https://relay.example.com', 'wss://relay.example.com'],
    ['https://relay.example.com/', 'wss://relay.example.com'],
    ['https://relay.example.com/base/', 'wss://relay.example.com/base'],
    ['https://relay.example.com:8443', 'wss://relay.example.com:8443'],
    ['wss://relay.example.com', 'wss://relay.example.com'],
    ['wss://relay.example.com//', 'wss://relay.example.com'],
    ['ws://relay.example.com', null],
    ['http://relay.example.com', null],
    ['http://localhost:8787', 'ws://localhost:8787'],
    ['ws://127.0.0.1:9', 'ws://127.0.0.1:9'],
    ['http://[::1]:8787', 'ws://[::1]:8787'],
    ['ftp://relay.example.com', null],
    ['not a url', null],
    ['', null],
  ])('%j -> %j', (input, expected) => {
    expect(toRelayWsUrl(input)).toBe(expected);
  });

  it('treats the https and wss spellings of one relay as equal', () => {
    expect(toRelayWsUrl('https://relay.example')).toBe(
      toRelayWsUrl('wss://relay.example/'),
    );
  });

  it('exports the loopback hosts', () => {
    expect([...LOOPBACK_HOSTS].sort()).toEqual(
      ['127.0.0.1', '[::1]', 'localhost'].sort(),
    );
  });
});
