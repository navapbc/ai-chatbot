import { getParticipantSource, summarize } from '@/lib/data/participant-source';
import { guardRelayRoute } from '@/lib/relay/route-guard';

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ recordId: string }> },
) {
  const guard = await guardRelayRoute();
  if (!guard.ok) return noStore(guard.response);

  const { recordId } = await params;
  const participant = await getParticipantSource().get(recordId);
  if (!participant) {
    return noStore(
      Response.json(
        { code: 'not_found:api', message: 'Participant not found.' },
        { status: 404 },
      ),
    );
  }

  return noStore(Response.json(summarize(participant)));
}
