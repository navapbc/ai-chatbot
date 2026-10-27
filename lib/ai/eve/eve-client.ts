import type { InputAnswer } from '@/lib/ai/eve/input-requests';
import { internalSecret } from '@/lib/relay/internal-secret';
import type { RelayRunRef } from '@/lib/relay/eve-relay-browser';

// Eve is mounted on this app's own origin: next.config.ts wraps the config in
// `withEve()`, which rewrites /eve/v1/** to the Eve runtime. So the default
// target is our own loopback address, and the rewrite does the rest.
//
// It used to default to 127.0.0.1:2000, the port a standalone `eve dev` listens
// on. That is wrong under withEve(): in dev the child server takes an EPHEMERAL
// port (55788 on one run here), and in the container it is on 4274 — nothing
// ever listens on 2000, so every turn failed with ECONNREFUSED. Addressing our
// own origin works in both, without needing to know Eve's port.
//
// EVE_SERVER_URL still overrides, for pointing the adapter at an Eve server you
// run yourself.
const EVE_URL =
  process.env.EVE_SERVER_URL ?? `http://127.0.0.1:${process.env.PORT ?? 3000}`;

async function postJson(
  path: string,
  body: unknown,
  extraHeaders?: Record<string, string>,
) {
  const res = await fetch(`${EVE_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(extraHeaders ?? {}) },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(
      `Eve server ${path} responded ${res.status}: ${await res.text().catch(() => '')}`,
    );
  }
  const sessionId = res.headers.get('x-eve-session-id') ?? '';
  const json = (await res.json().catch(() => ({}))) as {
    continuationToken?: string;
  };
  return { sessionId, continuationToken: json.continuationToken ?? '' };
}

export async function createEveSession(
  message: string,
  model?: string,
  relay?: RelayRunRef,
) {
  // The run reference travels in headers only (never the body or message
  // text); the channel auth turns it into session attributes for the browser
  // tool. It is not secret: no token is sent, the tool fetches one later.
  const headers: Record<string, string> = {};
  if (model) headers['x-eve-model'] = model;
  if (relay) {
    // Without the shared secret the channel would ignore the reference, so
    // fail here rather than send it anyway.
    const secret = internalSecret();
    if (!secret) throw new Error('relay sessions require EVE_INTERNAL_SECRET');
    headers['x-eve-internal-secret'] = secret;
    headers['x-eve-relay-url'] = relay.url;
    headers['x-eve-relay-run'] = relay.runId;
  }
  const { sessionId, continuationToken } = await postJson(
    '/eve/v1/session',
    { message },
    Object.keys(headers).length > 0 ? headers : undefined,
  );
  // Both are stored on the run row; an empty one would strand the chat.
  if (!sessionId || !continuationToken) {
    throw new Error('Eve create returned no session id or token');
  }
  return { sessionId, continuationToken };
}

/**
 * Continue a session with a plain message (string form) or with structured
 * answers to pending `ask_question` requests. Only keys that are set are sent,
 * so an answer turn carries no `message`.
 *
 * Returns the session the turn landed on (`x-eve-session-id`). When the old
 * run has ended, Eve's send path starts a fresh session instead of failing,
 * so callers must compare it with the session they meant to continue.
 */
export async function continueEveSession(
  sessionId: string,
  continuationToken: string,
  input: string | { message?: string; inputResponses?: InputAnswer[] },
) {
  const { message, inputResponses } =
    typeof input === 'string'
      ? { message: input, inputResponses: undefined }
      : input;
  const { continuationToken: next, sessionId: landedOn } = await postJson(
    `/eve/v1/session/${sessionId}`,
    {
      continuationToken,
      ...(message !== undefined ? { message } : {}),
      ...(inputResponses !== undefined ? { inputResponses } : {}),
    },
  );
  if (!landedOn) throw new Error('Eve continue returned no session id');
  return { continuationToken: next, sessionId: landedOn };
}

// Eve session ids are `wrun_` plus a ULID; this admits only a safe path
// segment, so a stored id can never redirect the request.
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{1,200}$/;

export type CancelTurnStatus = 'accepted' | 'no_active_turn';

/**
 * Cancel the session's in-flight turn. Both of Eve's statuses are success
 * ("no_active_turn" covers a settled turn or a duplicate cancel). The session
 * keeps its context; callers that must refuse further turns do so themselves.
 *
 * Same conventions as `continueEveSession`: no relay or internal-secret
 * headers (those are a session-creation grant; Eve's cancel route runs the
 * channel's route auth only). Errors carry no session id and no response
 * text.
 */
export async function cancelEveTurn(
  sessionId: string,
): Promise<CancelTurnStatus> {
  if (!SESSION_ID_PATTERN.test(sessionId)) {
    throw new Error('Eve cancel: invalid session id');
  }
  const res = await fetch(`${EVE_URL}/eve/v1/session/${sessionId}/cancel`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{}',
  });
  if (!res.ok) throw new Error(`Eve cancel responded ${res.status}`);
  const json = (await res.json().catch(() => null)) as {
    status?: unknown;
  } | null;
  const status = json?.status;
  if (status !== 'accepted' && status !== 'no_active_turn') {
    throw new Error('Eve cancel returned an unknown status');
  }
  return status;
}

/**
 * Read a session's durable event stream from `startIndex` (an absolute event
 * count, so it is also the number of events already consumed).
 *
 * Always send it explicitly: omitting the parameter makes Eve replay from event
 * 0, which on a follow-up turn re-delivers the previous turn and its
 * `session.waiting` boundary. Passing the cursor also makes the read immune to
 * the gap between starting a turn and subscribing — events emitted in between
 * are still waiting at that index rather than being missed.
 */
export async function openEveStream(
  sessionId: string,
  startIndex = 0,
  signal?: AbortSignal,
): Promise<Response> {
  const res = await fetch(
    `${EVE_URL}/eve/v1/session/${sessionId}/stream?startIndex=${startIndex}`,
    { signal },
  );
  if (!res.ok || !res.body) {
    throw new Error(`Eve stream ${sessionId} responded ${res.status}`);
  }
  return res;
}

export async function* parseNdjson(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<any> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl = buf.indexOf('\n');
      while (nl !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) {
          try {
            yield JSON.parse(line);
          } catch {
            /* skip malformed line */
          }
        }
        nl = buf.indexOf('\n');
      }
    }
    const tail = buf.trim();
    if (tail) {
      try {
        yield JSON.parse(tail);
      } catch {
        /* ignore */
      }
    }
  } finally {
    reader.releaseLock();
  }
}
