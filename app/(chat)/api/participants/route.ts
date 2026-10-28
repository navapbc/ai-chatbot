import { getParticipantSource } from '@/lib/data/participant-source';
import { ChatSDKError } from '@/lib/errors';
import { readBoundedText } from '@/lib/relay/read-bounded-body';
import { guardRelayRoute } from '@/lib/relay/route-guard';

const DEFAULT_LIMIT = 10;
const MAX_LIMIT = 20;
const MAX_QUERY_LENGTH = 100;
const MAX_BODY_BYTES = 4 * 1024;

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};

// The search term travels in a POST body, never the URL, so a participant name
// cannot land in an access or server log. Nothing here logs the term.
export async function POST(request: Request) {
  const guard = await guardRelayRoute(request, { requireOrigin: true });
  if (!guard.ok) return noStore(guard.response);

  const badRequest = () =>
    noStore(new ChatSDKError('bad_request:api').toResponse());

  let body: unknown;
  try {
    const read = await readBoundedText(request, MAX_BODY_BYTES);
    if (!read.ok) return badRequest();
    body = JSON.parse(read.text);
  } catch {
    return badRequest();
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return badRequest();
  }

  const { q: rawQ, limit: rawLimit } = body as Record<string, unknown>;
  if (rawQ !== undefined && typeof rawQ !== 'string') return badRequest();
  const q = (rawQ ?? '').trim();
  if (q.length > MAX_QUERY_LENGTH) return badRequest();

  let limit = DEFAULT_LIMIT;
  if (rawLimit !== undefined) {
    if (
      typeof rawLimit !== 'number' ||
      !Number.isInteger(rawLimit) ||
      rawLimit < 0
    ) {
      return badRequest();
    }
    limit = Math.min(MAX_LIMIT, Math.max(1, rawLimit));
  }

  const participants = await getParticipantSource().search(q, limit);
  return noStore(Response.json({ participants }));
}
