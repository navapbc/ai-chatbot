import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  type PendingInputRequest,
  validateAnswers,
} from '../../lib/ai/eve/input-requests';
import {
  mustRefuseRelayText,
  stripSensitiveText,
} from '../../lib/data/sensitive-record';
import { sensitiveAnswer } from '../../extension/panel/lib/question-kind';
import {
  type Draft,
  markFilled,
  NeedsYouCard,
} from '../../extension/panel/screens/needs-you';

// Free-text sensitive questions on the Needs You card. The behaviors that need
// a DOM (pressing Fill it in) run in the panel-shell e2e.

const flagged: PendingInputRequest = {
  requestId: 'r1',
  prompt: 'What is the applicant Social Security number?',
  display: 'text',
  options: [],
  allowFreeform: true,
};
const ordinary: PendingInputRequest = {
  requestId: 'r2',
  prompt: 'What county does the applicant live in?',
  display: 'text',
  options: [],
  allowFreeform: true,
};

const html = (pending: PendingInputRequest[]) =>
  renderToStaticMarkup(
    createElement(NeedsYouCard, {
      runId: 'run-1',
      pending,
      sendAnswers: async () => ({ ok: true as const }),
    }),
  );

describe('a flagged free-text question', () => {
  it('renders the masked typing card, no text box and no dead end', () => {
    const out = html([flagged]);
    expect(out).toContain('type="password"');
    expect(out).toContain('Fill it in');
    expect(out).toContain('I typed it in the page');
    expect(out).not.toContain('<textarea');
    expect(out).not.toContain('cannot be answered here');
  });

  it('answers with exactly the text provided, which the server accepts', () => {
    const answer = sensitiveAnswer(flagged, 'provided');
    expect(answer).toEqual({ requestId: 'r1', text: 'provided' });
    expect(validateAnswers([flagged], [answer])).toEqual({
      ok: true,
      responses: [{ requestId: 'r1', text: 'provided' }],
    });
    // The route's prose guard, with the question's prompt in front.
    for (const text of ['provided', 'skip']) {
      const withPrompt = stripSensitiveText(`${flagged.prompt}: ${text}`);
      expect(mustRefuseRelayText(withPrompt)).toBe(false);
      expect(stripSensitiveText(text)).toBe(text);
    }
    const skip = sensitiveAnswer(flagged, 'skip');
    expect(validateAnswers([flagged], [skip]).ok).toBe(true);
  });
});

describe('an ordinary free-text question', () => {
  it('keeps its text box and has no typing card', () => {
    const out = html([ordinary]);
    expect(out).toContain('<textarea');
    expect(out).not.toContain('type="password"');
  });
});

describe('markFilled', () => {
  it('records provided and filled', () => {
    const next = markFilled(new Map<string, Draft>(), 'r1');
    expect(next.get('r1')).toEqual({
      kind: 'sensitive',
      choice: 'provided',
      filled: true,
    });
  });

  it('keeps an explicit Skip pressed while the fill was in flight', () => {
    const current = new Map<string, Draft>([
      ['r1', { kind: 'sensitive', choice: 'skip' }],
    ]);
    expect(markFilled(current, 'r1').get('r1')).toEqual({
      kind: 'sensitive',
      choice: 'skip',
    });
  });

  it('replaces I typed it in the page', () => {
    const current = new Map<string, Draft>([
      ['r1', { kind: 'sensitive', choice: 'provided' }],
    ]);
    expect(markFilled(current, 'r1').get('r1')).toMatchObject({
      filled: true,
    });
  });
});
