'use client';

import { motion } from 'framer-motion';
import { Button } from './ui/button';
import { memo } from 'react';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { VisibilityType } from './visibility-selector';
import type { ChatMessage } from '@/lib/types';

interface SuggestedActionsLocalProps {
  chatId: string;
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  selectedVisibilityType: VisibilityType;
}

// Local/preview environments don't have Apricot API access, so these
// suggestions carry the full participant JSON inline instead of a record ID
// the agent would otherwise look up via Apricot.
function PureSuggestedActionsLocal({
  chatId,
  sendMessage,
  selectedVisibilityType,
}: SuggestedActionsLocalProps) {
  const suggestedActions = [
    {
      title: 'Fill out IHSS for Celeste Thomas II',
      label: 'riversideihss.org/IntakeApp',
      action:
        'fill out IHSS https://riversideihss.org/IntakeApp for\n\n{ "record": { "record_id": "339619", "file_open_date": "2025-10-23", "dpss_referral_date": "2025-10-22", "verbal_consent_provided": true }, "participant": { "name": { "first": "Celeste", "middle": "NAVA", "last": "Thomas II" }, "date_of_birth": "2000-01-02", "participant_type": "Parent", "funding_source": "FED", "calworks_id": "A100103", "ethnicity": "Hispanic/Latino", "gender": "Female", "primary_language": "English", "special_needs": false, "farm_worker": false, "marital_status": "Single parent household" }, "contact_information": { "preferred_method": null, "phones": { "home": "555-555-5555", "work": "666-666-6666", "cell": "777-777-7777", "main": "888-888-8888" }, "email": "testnava@email.com" }, "address": { "residential": { "street": "5556 Test Blvd", "unit": "Apt 556", "city": "WILDOMAR", "state": "California", "county": "Riverside", "zip": "92595" }, "mailing": { "street": "5556 Test Blvd", "unit": "Apt 556", "city": "WILDOMAR", "state": "California", "county": "Riverside", "zip": "92595" } }, "family_profile": { "linked": false, "notes": "No existing Family Profile linked at time of record" } } }',
    },
    {
      title: 'Fill out WIC for Amelie Thomas I',
      label: 'ruhealth.org/apply-4-wic-form',
      action:
        'Fill out WIC https://www.ruhealth.org/appointments/apply-4-wic-form# for { "record_id": "338618", "participant": { "name": { "first": "Amelie", "middle": "NAVA", "last": "Thomas I" }, "date_of_birth": "2000-01-01", "participant_type": "Parent", "funding_source": "Fed", "calworks_id": "A100102", "ethnicity": "Hispanic/Latino", "primary_language": "English", "gender": "Female", "special_needs": false, "marital_status": "Single parent household" }, "dates": { "file_open_date": "2025-10-23", "dpss_referral_date": "2025-10-22" }, "contact_information": { "phones": { "home": "5555555555", "work": "6666666666", "cell": "7777777777", "main": "8888888888" }, "email": "testnava@email.com" }, "address": { "residential": { "street": "5555 Test Blvd", "unit": "Apt 555", "city": "BANNING", "state": "CA", "county": "Riverside", "zip": "92220", "country": "US" }, "mailing": { "street": "5555 Test Blvd", "unit": "Apt 555", "city": "BANNING", "state": "CA", "county": "Riverside", "zip": "92220", "country": "US" } } }',
    },
    {
      title: 'Fill out SNAP for Sawyer Thomas XX',
      label: 'BenefitsCal.com',
      action:
        'Fill out SNAP for BenefitsCal.com for { "record_id": "339637", "participant": { "name": { "first": "Sawyer", "middle": "NAVA", "last": "Thomas XX" }, "date_of_birth": "1954-01-10", "participant_type": "Other", "funding_source": "Fed", "calworks_id": "A100121", "ethnicity": "Hispanic/Latino", "primary_language": "Spanish", "gender": "Male", "special_needs": false, "marital_status": "Other" }, "dates": { "file_open_date": "2025-10-23", "dpss_referral_date": "2025-10-22" }, "contact_information": { "phones": { "home": "5555555555", "work": null, "cell": "7777777777", "main": "8888888888" }, "email": "testnava@email.com" }, "address": { "residential": { "street": "5574 Test Blvd", "unit": "Apt 574", "city": "WILDOMAR", "state": "CA", "county": "Riverside", "zip": "92505", "country": "US" }, "mailing": { "street": "5574 Test Blvd", "unit": "Apt 574", "city": "WILDOMAR", "state": "CA", "county": "Riverside", "zip": "92505", "country": "US" } } }',
    },
  ];

  return (
    <div
      data-testid="suggested-actions-local"
      className="grid sm:grid-cols-2 gap-2 w-full overflow-hidden"
    >
      {suggestedActions.map((suggestedAction, index) => (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ delay: 0.05 * index }}
          key={`suggested-action-local-${suggestedAction.title}-${index}`}
          className={`${index > 1 ? 'hidden sm:block' : 'block'} min-w-0`}
        >
          <Button
            variant="ghost"
            onClick={async () => {
              window.history.replaceState({}, '', `/chat/${chatId}`);

              sendMessage({
                role: 'user',
                parts: [{ type: 'text', text: suggestedAction.action }],
              });
            }}
            className="text-left border border-border rounded-lg px-3 py-1.5 text-xs font-medium w-full h-auto justify-start items-center transition-colors duration-200 bg-muted text-foreground hover:bg-accent hover:text-accent-foreground whitespace-nowrap overflow-hidden"
          >
            <span className="truncate max-w-full">{suggestedAction.title}</span>
          </Button>
        </motion.div>
      ))}
    </div>
  );
}

export const SuggestedActionsLocal = memo(
  PureSuggestedActionsLocal,
  (prevProps, nextProps) => {
    if (prevProps.chatId !== nextProps.chatId) return false;
    if (prevProps.selectedVisibilityType !== nextProps.selectedVisibilityType)
      return false;

    return true;
  },
);
