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

export async function createEveSession(message: string, model?: string) {
  const { sessionId, continuationToken } = await postJson(
    '/eve/v1/session',
    { message },
    model ? { 'x-eve-model': model } : undefined,
  );
  return { sessionId, continuationToken };
}

export async function continueEveSession(
  sessionId: string,
  continuationToken: string,
  message: string,
) {
  const { continuationToken: next } = await postJson(
    `/eve/v1/session/${sessionId}`,
    { continuationToken, message },
  );
  return { continuationToken: next };
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
