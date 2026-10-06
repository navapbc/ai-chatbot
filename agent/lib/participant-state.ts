// Durable per-session slot holding the participant record for the Eve agent.
//
// Why this exists at all: the legacy route binds the record into its tools at
// wiring time (`createGapAnalysisTool(participant)` in the chat route). Eve
// tools cannot work that way — they are module-level `defineTool` default
// exports discovered by path, with no construction step to pass an argument
// through. Nor can a tool find the record itself: `ToolContext` carries the
// session, its sandbox and its skills, but no message history.
//
// So the record is put into session state by a hook the moment a message
// arrives (agent/hooks/participant.ts) and read back out by the tools. State
// from `defineState` is durable and does not reset between turns, which is
// what makes this survive the compaction that would otherwise drop the
// opening message carrying the record.
//
// Declared here rather than in a tool file so the hook and every tool share
// one slot. `agent/lib/` is eve's documented home for code shared by agent
// files.

import { defineState } from 'eve/context';
import type { Participant } from '@/lib/data/participants';

export const participantState = defineState<Participant | null>(
  'labs-asp.participant',
  () => null,
);
