import { eveChannel } from 'eve/channels/eve';
import {
  ForbiddenError,
  isLoopbackRequest,
  type AuthFn,
} from 'eve/channels/auth';
import { secretMatches } from '../../../lib/ai/eve/loopback-attributes';
const internal: AuthFn<Request> = (request) => {
  if (
    !isLoopbackRequest(request) ||
    !secretMatches(
      request.headers.get('x-eve-internal-secret'),
      process.env.EVE_INTERNAL_SECRET,
    )
  ) {
    throw new ForbiddenError({
      code: 'gsn_refused',
      message: 'Internal service authentication required.',
    });
  }
  return {
    subject: 'gsn-preparation',
    principalId: 'gsn-preparation',
    principalType: 'service',
    authenticator: 'gsn-internal',
    attributes: {},
  };
};
export default eveChannel({ auth: [internal] });
