const EVE_URL = process.env.EVE_SERVER_URL ?? 'http://127.0.0.1:2000';

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
