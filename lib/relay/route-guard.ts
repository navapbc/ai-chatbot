import { auth } from '@/app/(auth)/auth';
import { ChatSDKError } from '@/lib/errors';
import { originRefusal } from '@/lib/relay/request-origin';

// Guest accounts are created by app/(auth)/auth.ts with this email shape.
const GUEST_EMAIL_PATTERN = /^guest-[0-9a-f]{8}@preview\.local$/i;

export const isGuestEmail = (email: string | null | undefined): boolean =>
  typeof email === 'string' && GUEST_EMAIL_PATTERN.test(email);

export type RelayRouteGuard =
  | { ok: true; userId: string }
  | { ok: false; response: Response };

// Shared entry check for relay routes: flag first (bodyless 404, before auth),
// then the extension origin (when RELAY_EXTENSION_ORIGIN is set and a request
// is passed), then a signed-in non-guest user. A foreign origin gets a 403
// before auth, so an unauthenticated caller learns nothing more.
export const guardRelayRoute = async (
  request?: Request,
  opts: { requireOrigin?: boolean } = {},
): Promise<RelayRouteGuard> => {
  if (process.env.RELAY_RUNS_ENABLED !== '1') {
    return { ok: false, response: new Response(null, { status: 404 }) };
  }

  if (request && originRefusal(request, opts) !== null) {
    return {
      ok: false,
      response: new ChatSDKError('forbidden:chat').toResponse(),
    };
  }

  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) {
    return {
      ok: false,
      response: new ChatSDKError('unauthorized:chat').toResponse(),
    };
  }

  if (isGuestEmail(session.user?.email)) {
    return {
      ok: false,
      response: new ChatSDKError('forbidden:chat').toResponse(),
    };
  }

  return { ok: true, userId };
};
