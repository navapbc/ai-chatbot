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

  it('treats a prompt that isSensitiveField flags as sensitive without those options', () => {
    expect(
      classifyQuestion(req({ prompt: 'Social Security number' })),
    ).toBe('sensitive');
    expect(
      classifyQuestion(
        req({
          prompt: 'Bank account number',
          options: [{ id: 'a', label: 'A' }],
          allowFreeform: false,
        }),
      ),
    ).toBe('sensitive');
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

  it('falls back to the fixed literal text when only free text is allowed', () => {
    const r = req({ prompt: 'Social Security number' });
    expect(sensitiveAnswer(r, 'provided')).toEqual({
      requestId: 'r1',
      text: 'provided',
    });
    expect(sensitiveAnswer(r, 'skip')).toEqual({
      requestId: 'r1',
      text: 'skip',
    });
  });

  it('cannot answer a flagged question with other options and no free text', () => {
    const r = req({
      prompt: 'Bank account number',
      options: [{ id: 'a', label: 'A' }],
      allowFreeform: false,
    });
    expect(sensitiveAnswer(r, 'provided')).toBeNull();
  });

  it('never throws on malformed requests', () => {
    expect(sensitiveAnswer(null as unknown as PendingInputRequest, 'skip')).toBe(
      null,
    );
    expect(
      sensitiveAnswer({ requestId: '' } as PendingInputRequest, 'skip'),
    ).toBe(null);
  });
});
