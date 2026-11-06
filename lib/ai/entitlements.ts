import type { UserType } from '@/app/(auth)/auth';
import { selectableChatModelIds, type ChatModel } from './models';

interface Entitlements {
  maxMessagesPerDay: number;
  availableChatModelIds: Array<ChatModel['id']>;
}

export const entitlementsByUserType: Record<UserType, Entitlements> = {
  /*
   * For users with an account
   */
  regular: {
    maxMessagesPerDay: 500,
    // Derived from `chatModels` rather than listed by hand, so adding a model
    // to the picker can't silently leave it filtered out here. `chatModels`
    // already excludes the dev-only overrides in production.
    availableChatModelIds: selectableChatModelIds,
  },

  /*
   * TODO: For users with an account and a paid membership
   */
};
