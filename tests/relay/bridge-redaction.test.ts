import { describe, expect, it } from 'vitest';
import { createRedactor, type Redactor } from '../../lib/relay/redactor';
import {
  REDACTION_FAILED,
  redactErrorMessage,
  redactEventParams,
  redactReply,
} from '../../extension/src/redaction';

// The bridge's fail-closed wrappers, against stub redactors that misbehave.

const stub = (overrides: Partial<Redactor>): Redactor => ({
  addKnownValue: () => true,
  hasKnownValue: () => false,
  clear: () => {},
  redactResult: async (_m, r) => r,
  redactEvent: (_m, p) => p,
  ...overrides,
});

const RAW = { result: { value: 'raw page data' } };

describe('redactReply', () => {
  it('returns the redacted result', async () => {
    const redactor = stub({ redactResult: async () => ({ result: 'safe' }) });
    await expect(
      redactReply(redactor, 'Runtime.evaluate', RAW),
    ).resolves.toEqual({ result: 'safe' });
  });

  it('fails with an error, never the raw result, when the redactor rejects', async () => {
    const redactor = stub({
      redactResult: async () => {
        throw new Error('boom');
      },
    });
    await expect(
      redactReply(redactor, 'Runtime.evaluate', RAW),
    ).rejects.toThrow(REDACTION_FAILED);
  });

  it('fails when the redactor throws synchronously', async () => {
    const redactor = stub({
      redactResult: (() => {
        throw new Error('boom');
      }) as Redactor['redactResult'],
    });
    await expect(
      redactReply(redactor, 'Runtime.evaluate', RAW),
    ).rejects.toThrow(REDACTION_FAILED);
  });

  it('fails when the redactor answers null for a non-null result', async () => {
    const redactor = stub({ redactResult: async () => null });
    await expect(
      redactReply(redactor, 'DOM.describeNode', RAW),
    ).rejects.toThrow(REDACTION_FAILED);
  });

  it('passes the describe function through', async () => {
    const describe = async () => ({ inputType: 'password' });
    let seen: unknown;
    const redactor = stub({
      redactResult: async (_m, r, d) => {
        seen = d;
        return r;
      },
    });
    await redactReply(redactor, 'Accessibility.getFullAXTree', RAW, describe);
    expect(seen).toBe(describe);
  });
});

describe('redactEventParams', () => {
  it('drops the event when the redactor throws', () => {
    const redactor = stub({
      redactEvent: () => {
        throw new Error('boom');
      },
    });
    expect(redactEventParams(redactor, 'Runtime.consoleAPICalled', RAW)).toBe(
      null,
    );
  });

  it('drops the event when the redactor answers null', () => {
    const redactor = stub({ redactEvent: () => null });
    expect(redactEventParams(redactor, 'Runtime.consoleAPICalled', RAW)).toBe(
      null,
    );
  });
});

describe('redactErrorMessage', () => {
  it('scrubs a known value from an error message', () => {
    const redactor = createRedactor();
    redactor.addKnownValue('900-77-4321');
    const out = redactErrorMessage(redactor, 'no node matches 900774321');
    expect(out).not.toContain('900774321');
    expect(out).toContain('[masked]');
  });

  it('falls back to a generic message when scrubbing fails', () => {
    const redactor = stub({
      redactEvent: () => {
        throw new Error('boom');
      },
    });
    expect(redactErrorMessage(redactor, 'anything')).toBe('command failed');
    expect(
      redactErrorMessage(stub({ redactEvent: () => null }), 'anything'),
    ).toBe('command failed');
  });
});
