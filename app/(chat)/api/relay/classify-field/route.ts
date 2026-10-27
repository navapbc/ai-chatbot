import { getRunStore } from '@/lib/ai/eve/run-store';
import { ChatSDKError } from '@/lib/errors';
import { askJev, isJevEnabled } from '@/lib/jev/client';
import {
  DESCRIPTOR_KEYS,
  describeFieldForJev,
  sensitiveFieldIsSensitive,
  sensitiveFieldQuestions,
} from '@/lib/jev/questions';
import { readBoundedText } from '@/lib/relay/read-bounded-body';
import { guardRelayRoute } from '@/lib/relay/route-guard';
import {
  type FieldDescriptor,
  isSensitiveField,
} from '@/lib/relay/sensitive-fields';

// Runtime classification of form fields the deterministic rules (in the
// extension) did not catch. The caller sends field descriptors only; this
// route never sees a typed value, and nothing here logs a descriptor or a
// verdict. Every failure resolves to "sensitive".

const MAX_BODY_BYTES = 64 * 1024;
const MAX_FIELDS = 50;
const MAX_CONCURRENCY = 8;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};
const badRequest = () =>
  noStore(new ChatSDKError('bad_request:api').toResponse());
const notFound = () => noStore(new Response(null, { status: 404 }));

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

// Only string values under the descriptor keys survive; everything else
// (value, url, text, non-string junk) is dropped before any classifier sees it.
const toDescriptor = (raw: Record<string, unknown>): FieldDescriptor => {
  const out: FieldDescriptor = {};
  for (const key of DESCRIPTOR_KEYS) {
    const v = raw[key];
    if (typeof v === 'string') out[key] = v;
  }
  return out;
};

const classify = async (descriptor: FieldDescriptor): Promise<boolean> => {
  try {
    if (isSensitiveField(descriptor)) return true;
    if (Object.keys(describeFieldForJev(descriptor)).length === 0) return true;
    const { state, questions } = sensitiveFieldQuestions(descriptor);
    const outcome = await askJev({
      state,
      questions,
      feature: 'sensitive-field',
    });
    return sensitiveFieldIsSensitive(outcome);
  } catch {
    return true;
  }
};

// Order-preserving map with at most `limit` calls in flight.
const mapLimited = async <T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> => {
  const results = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i]);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, worker),
  );
  return results;
};

export async function POST(request: Request) {
  const guard = await guardRelayRoute(request, { requireOrigin: true });
  if (!guard.ok) return noStore(guard.response);

  let body: unknown;
  try {
    const read = await readBoundedText(request, MAX_BODY_BYTES);
    if (!read.ok) return badRequest();
    body = JSON.parse(read.text);
  } catch {
    return badRequest();
  }

  if (!isRecord(body)) return badRequest();
  const { runId, fields } = body;
  if (typeof runId !== 'string' || !UUID.test(runId)) return badRequest();
  if (
    !Array.isArray(fields) ||
    fields.length < 1 ||
    fields.length > MAX_FIELDS ||
    !fields.every(isRecord)
  ) {
    return badRequest();
  }

  let run: Awaited<ReturnType<ReturnType<typeof getRunStore>['get']>>;
  try {
    run = await getRunStore().get(guard.userId, runId);
  } catch {
    return noStore(new ChatSDKError('offline:chat').toResponse());
  }
  // An ended (stopped) run looks like an unknown one, as on the token routes.
  if (!run || run.relayOrigin === null || run.endedAt !== null) {
    return notFound();
  }

  const descriptors = (fields as Record<string, unknown>[]).map(toDescriptor);

  if (!isJevEnabled('sensitive-field')) {
    return noStore(Response.json({ sensitive: descriptors.map(() => true) }));
  }

  const sensitive = await mapLimited(descriptors, MAX_CONCURRENCY, classify);
  return noStore(Response.json({ sensitive }));
}
