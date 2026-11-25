import { Preparation } from '@/lib/gsn/contract';
import { verifyPreparation } from '@/lib/gsn/signature';
import { claimRequest, finishRequest } from '@/lib/gsn/replays';
import { parseNdjson } from '@/lib/ai/eve/eve-client';
import { internalSecret } from '@/lib/relay/internal-secret';
export const maxDuration = 240;
const runtime = 'http://127.0.0.1:4275';
export async function POST(req: Request) {
  const secret = internalSecret();
  if (!process.env.GSN_BRIDGE_PUBLIC_KEY || !secret)
    return Response.json({ error: 'not_configured' }, { status: 503 });
  if (Number(req.headers.get('content-length') ?? 0) > 65536)
    return new Response(null, { status: 413 });
  const reader = req.body?.getReader();
  let body = '';
  if (!reader) return new Response(null, { status: 400 });
  const decoder = new TextDecoder();
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      body += decoder.decode(chunk.value, { stream: true });
      if (body.length > 65536) {
        await reader.cancel();
        return new Response(null, { status: 413 });
      }
    }
    body += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  if (!verifyPreparation(body, req.headers, process.env.GSN_BRIDGE_PUBLIC_KEY))
    return new Response(null, { status: 401 });
  let input: Preparation;
  try {
    input = Preparation.parse(JSON.parse(body));
  } catch {
    return new Response(null, { status: 400 });
  }
  if (input.requestId !== req.headers.get('x-gsn-nonce'))
    return new Response(null, { status: 400 });
  const claim = await claimRequest(input.requestId);
  if (claim === 'limited')
    return Response.json(
      { error: 'preparation_limit', saved: false },
      { status: 429 },
    );
  if (claim === 'duplicate')
    return Response.json(
      {
        error: 'request_already_started',
        recovery:
          'Read the application before starting a new preparation. No application write is performed by this service.',
      },
      { status: 409 },
    );
  const headers = {
    'content-type': 'application/json',
    'x-eve-internal-secret': secret,
  };
  const signal = AbortSignal.timeout(210000);
  let stream: Response | undefined;
  let sessionId: string | null = null;
  try {
    const created = await fetch(`${runtime}/eve/v1/session`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ message: JSON.stringify(input) }),
      signal,
    });
    sessionId = created.headers.get('x-eve-session-id');
    await created.body?.cancel();
    if (!created.ok || !sessionId || !/^[A-Za-z0-9_-]{1,200}$/.test(sessionId))
      throw new Error('runtime_unavailable');
    stream = await fetch(
      `${runtime}/eve/v1/session/${sessionId}/stream?startIndex=0`,
      { headers, signal },
    );
    if (!stream.ok || !stream.body) throw new Error('runtime_unavailable');
    for await (const event of parseNdjson(stream.body)) {
      const result = event.data?.result;
      if (
        event.type === 'action.result' &&
        result?.kind === 'tool-result' &&
        result.toolName === 'review_section'
      ) {
        const proposal = result.output as Record<string, unknown>;
        if (
          proposal?.format !== 'gsn-eve-jev-preparation-v1' ||
          proposal.requestId !== input.requestId
        )
          throw new Error('invalid_preparation');
        return Response.json(
          {
            ...proposal,
            eve: {
              mode: 'live',
              sessionId,
              runtime: 'riverside-eve-0.27',
              model: 'claude-sonnet-5-5',
            },
          },
          { headers: { 'Cache-Control': 'no-store' } },
        );
      }
      if (event.type === 'session.waiting') break;
    }
    throw new Error('no_preparation');
  } catch {
    return Response.json(
      { error: 'combined_preparation_failed', saved: false },
      { status: 502 },
    );
  } finally {
    await stream?.body?.cancel().catch(() => {});
    if (sessionId && /^[A-Za-z0-9_-]{1,200}$/.test(sessionId)) {
      await fetch(`${runtime}/eve/v1/session/${sessionId}/cancel`, {
        method: 'POST',
        headers,
        body: '{}',
        signal: AbortSignal.timeout(5000),
      })
        .then((response) => response.body?.cancel())
        .catch(() => {});
    }
    await finishRequest(input.requestId).catch(() => {});
  }
}
