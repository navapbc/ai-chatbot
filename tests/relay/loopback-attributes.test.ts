import { describe, expect, it } from 'vitest';
import { attributesFromHeaders } from '@/lib/ai/eve/loopback-attributes';

const SECRET = 'internal-secret-for-tests';
const RUN = '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11';
const relay = {
  'x-eve-relay-url': 'wss://relay.example',
  'x-eve-relay-run': RUN,
};
const h = (o: Record<string, string>) => new Headers(o);

describe('attributesFromHeaders', () => {
  it('grants relay attributes on loopback with the right secret', () => {
    expect(
      attributesFromHeaders(
        h({ ...relay, 'x-eve-internal-secret': SECRET }),
        true,
        SECRET,
      ),
    ).toEqual({ relayUrl: 'wss://relay.example', relayRunId: RUN });
  });

  it('ignores the retired token header entirely', () => {
    const SENTINEL = 'SENTINEL-RELAY-TOKEN-0123';
    const result = attributesFromHeaders(
      h({
        ...relay,
        'x-eve-relay-token': SENTINEL,
        'x-eve-internal-secret': SECRET,
      }),
      true,
      SECRET,
    );
    expect(result).toEqual({
      relayUrl: 'wss://relay.example',
      relayRunId: RUN,
    });
    expect(JSON.stringify(result)).not.toContain(SENTINEL);
    expect(
      attributesFromHeaders(
        h({
          'x-eve-relay-url': 'wss://relay.example',
          'x-eve-relay-token': SENTINEL,
          'x-eve-internal-secret': SECRET,
        }),
        true,
        SECRET,
      ),
    ).toBeNull();
  });

  it('refuses a run header that is not a UUID', () => {
    expect(
      attributesFromHeaders(
        h({
          ...relay,
          'x-eve-relay-run': 'not-a-uuid',
          'x-eve-internal-secret': SECRET,
        }),
        true,
        SECRET,
      ),
    ).toBeNull();
  });

  it('grants nothing to a non-loopback request, even with the secret', () => {
    expect(
      attributesFromHeaders(
        h({ ...relay, 'x-eve-model': 'm', 'x-eve-internal-secret': SECRET }),
        false,
        SECRET,
      ),
    ).toBeNull();
  });

  it('withholds relay attributes without the secret header', () => {
    expect(attributesFromHeaders(h(relay), true, SECRET)).toBeNull();
  });

  it('withholds relay attributes for a wrong secret', () => {
    expect(
      attributesFromHeaders(
        h({ ...relay, 'x-eve-internal-secret': 'nope' }),
        true,
        SECRET,
      ),
    ).toBeNull();
  });

  it.each([undefined, ''])(
    'never grants relay when the env secret is %j',
    (env) => {
      expect(
        attributesFromHeaders(
          h({ ...relay, 'x-eve-internal-secret': env ?? '' }),
          true,
          env,
        ),
      ).toBeNull();
    },
  );

  it.each([
    ['x-eve-relay-url', 'wss://relay.example'],
    ['x-eve-relay-run', RUN],
  ])('grants nothing for a lone %s', (name, value) => {
    expect(
      attributesFromHeaders(
        h({ [name]: value, 'x-eve-internal-secret': SECRET }),
        true,
        SECRET,
      ),
    ).toBeNull();
  });

  it('keeps the model header working on loopback without any secret', () => {
    expect(
      attributesFromHeaders(h({ 'x-eve-model': 'claude-x' }), true, undefined),
    ).toEqual({ eveModel: 'claude-x' });
  });

  it('grants the model but not relay when relay is unauthenticated', () => {
    expect(
      attributesFromHeaders(
        h({ ...relay, 'x-eve-model': 'claude-x' }),
        true,
        SECRET,
      ),
    ).toEqual({ eveModel: 'claude-x' });
  });
});
