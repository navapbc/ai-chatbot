import { describe, expect, it } from 'vitest';
import type { PendingInputRequest } from '../../lib/ai/eve/input-requests';
import {
  classifyQuestion,
  sensitiveAnswer,
} from '../../extension/panel/lib/question-kind';

const req = (over: Partial<PendingInputRequest> = {}): PendingInputRequest => ({
  requestId: 'r1',
  prompt: 'What county does the applicant live in?',
  display: 'text',
  options: [],
  allowFreeform: true,
  ...over,
});

const SENSITIVE_OPTIONS = [
  { id: 'provided', label: 'I entered it myself' },
  { id: 'skip', label: 'Skip' },
];

describe('classifyQuestion', () => {
  it("treats the skill's provided/skip shape as sensitive, in either order", () => {
    const shape = { display: 'select' as const, allowFreeform: false };
    expect(
      classifyQuestion(req({ ...shape, options: SENSITIVE_OPTIONS })),
    ).toBe('sensitive');
    expect(
      classifyQuestion(
        req({ ...shape, options: [...SENSITIVE_OPTIONS].reverse() }),
      ),
    ).toBe('sensitive');
  });

  it('is not the sensitive shape when free text is allowed or the ids differ', () => {
    expect(
      classifyQuestion(
        req({ options: SENSITIVE_OPTIONS, allowFreeform: true }),
      ),
    ).toBe('choice');
    expect(
      classifyQuestion(
        req({
          allowFreeform: false,
          options: [
            { id: 'provided', label: 'a' },
            { id: 'provided', label: 'b' },
          ],
        }),
      ),
    ).toBe('choice');
    expect(
      classifyQuestion(
        req({
          allowFreeform: false,
          options: [...SENSITIVE_OPTIONS, { id: 'other', label: 'Other' }],
        }),
      ),
    ).toBe('choice');
  });

  it('keeps the two-option shape sensitive with an innocuous prompt', () => {
    expect(
      classifyQuestion(
        req({
          prompt: 'The third field in Applicant Information',
          display: 'select',
          options: SENSITIVE_OPTIONS,
          allowFreeform: false,
        }),
      ),
    ).toBe('sensitive');
  });

  it('treats a free-text question whose prompt isSensitiveField flags as sensitive', () => {
    expect(
      classifyQuestion(req({ prompt: 'What is your Social Security number?' })),
    ).toBe('sensitive');
  });

  it('keeps a choice whose prompt only mentions a sensitive topic a choice', () => {
    for (const prompt of [
      'Does anyone in the household receive Social Security benefits?',
      'Does the applicant have a bank account?',
      'Bank account number',
    ]) {
      const yesNo = req({
        prompt,
        display: 'select',
        options: [
          { id: 'yes', label: 'Yes' },
          { id: 'no', label: 'No' },
        ],
        allowFreeform: false,
      });
      expect(classifyQuestion(yesNo)).toBe('choice');
      // Options with free text too: still answered by option id.
      expect(classifyQuestion({ ...yesNo, allowFreeform: true })).toBe(
        'choice',
      );
    }
  });

  it('classifies a plain choice and a plain text question', () => {
    expect(
      classifyQuestion(
        req({
          display: 'select',
          options: [
            { id: 'yes', label: 'Yes' },
            { id: 'no', label: 'No' },
          ],
          allowFreeform: false,
        }),
      ),
    ).toBe('choice');
    expect(classifyQuestion(req())).toBe('text');
  });

  it('treats allowFreeform:false with no options as a choice (never text)', () => {
    expect(classifyQuestion(req({ allowFreeform: false, options: [] }))).toBe(
      'choice',
    );
  });

  it('never throws on malformed requests and never yields text for them', () => {
    const bad: unknown[] = [
      null,
      undefined,
      42,
      'x',
      [],
      {},
      { options: 'nope' },
      { options: [null, 1, 'x'], allowFreeform: false },
      { prompt: 7, options: null },
      { prompt: { toString: () => 'ssn' } },
    ];
    for (const b of bad) {
      const kind = classifyQuestion(b as PendingInputRequest);
      expect(['sensitive', 'choice', 'text']).toContain(kind);
      expect(kind).not.toBe('text');
    }
  });

  it('handles hostile option ids without throwing', () => {
    const hostile = req({
      allowFreeform: false,
      options: [
        { id: '__proto__', label: 'x' },
        { id: 'constructor', label: 'y' },
      ],
    });
    expect(classifyQuestion(hostile)).toBe('choice');
    const weird = req({
      allowFreeform: false,
      options: [
        { id: { toString: () => 'provided' }, label: 'x' },
        { id: 'skip', label: 'y' },
      ] as unknown as PendingInputRequest['options'],
    });
    expect(classifyQuestion(weird)).toBe('choice');
  });
});

describe('sensitiveAnswer', () => {
  it('answers by option id when the request carries provided/skip', () => {
    const r = req({ options: SENSITIVE_OPTIONS, allowFreeform: false });
    expect(sensitiveAnswer(r, 'provided')).toEqual({
      requestId: 'r1',
      optionId: 'provided',
    });
    expect(sensitiveAnswer(r, 'skip')).toEqual({
      requestId: 'r1',
      optionId: 'skip',
    });
  });

  it('never sends the word as free text for a prompt-flagged text question', () => {
    const r = req({ prompt: 'Social Security number' });
    expect(classifyQuestion(r)).toBe('sensitive');
    expect(sensitiveAnswer(r, 'provided')).toBeNull();
    expect(sensitiveAnswer(r, 'skip')).toBeNull();
  });

  it('has no answer for a request without provided/skip options', () => {
    const r = req({
      prompt: 'Bank account number',
      options: [{ id: 'a', label: 'A' }],
      allowFreeform: false,
    });
    expect(sensitiveAnswer(r, 'provided')).toBeNull();
  });

  it('never throws on malformed requests', () => {
    expect(
      sensitiveAnswer(null as unknown as PendingInputRequest, 'skip'),
    ).toBe(null);
    expect(
      sensitiveAnswer({ requestId: '' } as PendingInputRequest, 'skip'),
    ).toBe(null);
  });
});
