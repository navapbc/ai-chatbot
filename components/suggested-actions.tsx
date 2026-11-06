'use client';

import {
  buildApplicationPrompt,
  getParticipantById,
} from '@/lib/data/participants';

import { Button } from './ui/button';
import type { ChatMessage } from '@/lib/types';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { VisibilityType } from './visibility-selector';
import { memo } from 'react';
import { motion } from 'framer-motion';

interface SuggestedActionsProps {
  chatId: string;
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  selectedVisibilityType: VisibilityType;
}

// Each suggestion names a participant from lib/data/participants and a program.
// The participant JSON is sent inline with the message — the agent has no
// participant database to look the record up in.
const SUGGESTED_ACTIONS = [
  {
    recordId: '339619',
    title: 'Fill out IHSS for Celeste Thomas II',
    site: 'riversideihss.org/IntakeApp',
    target: 'IHSS at https://riversideihss.org/IntakeApp',
  },
  {
    recordId: '338618',
    title: 'Fill out WIC for Amelie Thomas I',
    site: 'ruhealth.org/apply-4-wic-form',
    target: 'WIC at https://www.ruhealth.org/appointments/apply-4-wic-form#',
  },
  {
    recordId: '339637',
    title: 'Fill out SNAP for Sawyer Thomas XX',
    site: 'benefitscal.com',
    target: 'SNAP at https://benefitscal.com/',
  },
];

function PureSuggestedActions({ chatId, sendMessage }: SuggestedActionsProps) {
  return (
    <div
      data-testid="suggested-actions"
      className="grid sm:grid-cols-2 gap-2 w-full overflow-hidden"
    >
      {SUGGESTED_ACTIONS.map((suggestedAction, index) => {
        const participant = getParticipantById(suggestedAction.recordId);
        if (!participant) return null;

        const handleClick = () => {
          window.history.replaceState({}, '', `/chat/${chatId}`);

          sendMessage({
            role: 'user',
            parts: [
              {
                type: 'text',
                text: buildApplicationPrompt({
                  participant,
                  target: suggestedAction.target,
                }),
              },
            ],
          });
        };

        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ delay: 0.05 * index }}
            key={suggestedAction.recordId}
            className={`${index > 1 ? 'hidden sm:block' : 'block'} min-w-0`}
          >
            <Button
              variant="ghost"
              onClick={handleClick}
              className="flex-col items-start gap-0.5 text-left border border-border rounded-lg px-3 py-2 h-auto w-full transition-colors duration-200 bg-muted text-foreground hover:bg-accent hover:text-accent-foreground overflow-hidden"
            >
              <span className="truncate max-w-full text-xs font-medium">
                {suggestedAction.title}
              </span>
            </Button>
          </motion.div>
        );
      })}
    </div>
  );
}

export const SuggestedActions = memo(
  PureSuggestedActions,
  (prevProps, nextProps) => {
    if (prevProps.chatId !== nextProps.chatId) return false;
    if (prevProps.selectedVisibilityType !== nextProps.selectedVisibilityType)
      return false;

    return true;
  },
);
