import { getParticipantSource } from '@/lib/data/participant-source';
import { ChatSDKError } from '@/lib/errors';
import { guardRelayRoute } from '@/lib/relay/route-guard';

const DEFAULT_LIMIT = 10;
const MAX_LIMIT = 20;
const MAX_QUERY_LENGTH = 100;

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};

export async function GET(request: Request) {
  const guard = await guardRelayRoute();
  if (!guard.ok) return noStore(guard.response);

  const params = new URL(request.url).searchParams;
  const q = (params.get('q') ?? '').trim();
  const rawLimit = params.get('limit');

  if (q.length > MAX_QUERY_LENGTH) {
    return noStore(new ChatSDKError('bad_request:api').toResponse());
  }

  let limit = DEFAULT_LIMIT;
  if (rawLimit !== null) {
    if (!/^\d+$/.test(rawLimit)) {
      return noStore(new ChatSDKError('bad_request:api').toResponse());
    }
    limit = Math.min(MAX_LIMIT, Math.max(1, Number(rawLimit)));
  }

  const participants = await getParticipantSource().search(q, limit);
  return noStore(Response.json({ participants }));
}
