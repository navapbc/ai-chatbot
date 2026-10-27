import { getRelayConfig } from '@/lib/relay/config';
import { guardRelayRoute } from '@/lib/relay/route-guard';

const noStore = (response: Response): Response => {
  response.headers.set('Cache-Control', 'no-store');
  return response;
};

export async function GET() {
  const guard = await guardRelayRoute();
  if (!guard.ok) return noStore(guard.response);
  return noStore(Response.json(getRelayConfig()));
}
