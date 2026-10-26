import { describe, expect, it } from 'vitest';
import { PARTICIPANTS, buildApplicationPrompt } from '@/lib/data/participants';
import { extractParticipantFromText } from '@/lib/jev/participant';
import {
  buildExtensionRunMessage,
  EXTENSION_RUN_MARKER,
} from '@/lib/relay/run-prompt';

describe('buildExtensionRunMessage', () => {
  it('starts with the marker and ends with the original text unchanged', () => {
    const text = 'Fill this form\n\n  for Rosa  ';
    const out = buildExtensionRunMessage(text);
    expect(out.startsWith(EXTENSION_RUN_MARKER)).toBe(true);
    expect(out.endsWith(text)).toBe(true);
    expect(out).toContain('extension-run');
    expect(out).toContain('overrides your other instructions');
  });

  it('keeps the participant record parseable from the opening message', () => {
    const opening = buildApplicationPrompt({
      participant: PARTICIPANTS[0],
      target: 'a WIC application',
    });
    const found = extractParticipantFromText(buildExtensionRunMessage(opening));
    expect(found?.record_id).toBe(PARTICIPANTS[0].record_id);
  });
});
