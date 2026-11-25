import { generateKeyPairSync, randomUUID, sign } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
const { claim, finish } = vi.hoisted(() => ({
  claim: vi.fn(),
  finish: vi.fn(),
}));
vi.mock('../../lib/gsn/replays', () => ({
  claimRequest: claim,
  finishRequest: finish,
}));
import { POST } from '../../app/(chat)/api/gsn/prepare/route';
const keys = generateKeyPairSync('ed25519');
function request(signed = true) {
  const nonce = randomUUID();
  const time = String(Date.now());
  const body = JSON.stringify({
    requestId: nonce,
    applicationId: randomUUID(),
    expectedVersion: 1,
    section: 'personal',
    data: { firstName: 'Fictional' },
    guide: {},
  });
  const headers: Record<string, string> = {
    'content-type': 'application/json',
  };
  if (signed)
    Object.assign(headers, {
      'x-gsn-time': time,
      'x-gsn-nonce': nonce,
      'x-gsn-signature': sign(
        null,
        Buffer.from(`POST\n/api/gsn/prepare\n${time}\n${nonce}\n${body}`),
        keys.privateKey,
      ).toString('base64url'),
    });
  return new Request('https://dev.example/api/gsn/prepare', {
    method: 'POST',
    headers,
    body,
  });
}
beforeEach(() => {
  vi.stubEnv(
    'GSN_BRIDGE_PUBLIC_KEY',
    keys.publicKey
      .export({ type: 'spki', format: 'der' })
      .toString('base64url'),
  );
  vi.stubEnv('EVE_INTERNAL_SECRET', 'test-internal-only-32-character-secret');
  claim.mockResolvedValue('claimed');
  finish.mockResolvedValue(undefined);
  vi.stubGlobal('fetch', vi.fn());
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});
describe('signed GSN preparation route', () => {
  it('rejects unsigned calls before claiming a request or reaching a model', async () => {
    expect((await POST(request(false))).status).toBe(401);
    expect(claim).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });
  it('refuses consumed request IDs and cost limits before reaching a model', async () => {
    claim.mockResolvedValueOnce('duplicate').mockResolvedValueOnce('limited');
    expect((await POST(request())).status).toBe(409);
    expect((await POST(request())).status).toBe(429);
    expect(fetch).not.toHaveBeenCalled();
  });
  it('does not report live preparation when the runtime fails, and releases its lease', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('private provider detail'));
    const response = await POST(request());
    expect(response.status).toBe(502);
    expect(await response.text()).not.toContain('private provider detail');
    expect(finish).toHaveBeenCalledOnce();
  });
});

it('returns a completed preparation even when cancellation of the held-open event stream never settles', async () => {
  const req = request();
  const input = await req.clone().json();
  let releaseCancellation = () => {};
  const heldOpen = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          `${JSON.stringify({
            type: 'action.result',
            data: {
              result: {
                kind: 'tool-result',
                toolName: 'review_section',
                output: {
                  format: 'gsn-eve-jev-preparation-v1',
                  requestId: input.requestId,
                  data: input.data,
                  jev: { mode: 'live', model: 'mock-only' },
                },
              },
            },
          })}\n`,
        ),
      );
    },
    cancel() {
      return new Promise<void>((resolve) => {
        releaseCancellation = resolve;
      });
    },
  });
  vi.mocked(fetch)
    .mockResolvedValueOnce(
      Response.json({}, { headers: { 'x-eve-session-id': 'wrun_mock' } }),
    )
    .mockResolvedValueOnce(new Response(heldOpen))
    .mockResolvedValueOnce(Response.json({ status: 'accepted' }));
  const pending = POST(req);
  const outcome = await Promise.race([
    pending,
    new Promise<null>((resolve) => setTimeout(() => resolve(null), 250)),
  ]);
  releaseCancellation();
  const response = await pending;
  expect(
    outcome,
    'completed provider work must not wait for stream cancellation',
  ).not.toBeNull();
  expect(response.status).toBe(200);
  expect(finish).toHaveBeenCalledOnce();
});
