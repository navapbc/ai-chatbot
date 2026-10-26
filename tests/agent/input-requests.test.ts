import { describe, expect, it } from 'vitest';
import {
  parseInputRequests,
  validateAnswers,
  type PendingInputRequest,
} from '@/lib/ai/eve/input-requests';

const textReq: PendingInputRequest = {
  requestId: 'r1',
  prompt: 'Name?',
  display: 'text',
  options: [],
  allowFreeform: true,
};
const selectReq: PendingInputRequest = {
  requestId: 'r2',
  prompt: 'Color?',
  display: 'select',
  options: [
    { id: 'red', label: 'Red' },
    { id: 'green', label: 'Green' },
    { id: 'blue', label: 'Blue' },
  ],
  allowFreeform: false,
};
const pending = [textReq, selectReq];

describe('parseInputRequests', () => {
  it('parses the observed text and select shapes', () => {
    const raw = [
      {
        requestId: 'r1',
        prompt: 'Name?',
        display: 'text',
        allowFreeform: true,
        action: { kind: 'x' },
      },
      {
        requestId: 'r2',
        prompt: 'Color?',
        display: 'select',
        allowFreeform: false,
        action: {},
        options: [
          { id: 'red', label: 'Red', description: 'warm' },
          { id: 'green', label: 'Green' },
          { id: 'blue', label: 'Blue' },
        ],
      },
    ];
    expect(parseInputRequests(raw)).toEqual(pending);
  });

  it('drops entries without a requestId and non-array input', () => {
    expect(
      parseInputRequests([
        { prompt: 'x', display: 'text', allowFreeform: true },
        null,
        'str',
        textReq,
      ]),
    ).toEqual([textReq]);
    expect(parseInputRequests(undefined)).toEqual([]);
    expect(parseInputRequests({})).toEqual([]);
  });

  it('treats anything but allowFreeform === true as option-only', () => {
    const [r] = parseInputRequests([
      { requestId: 'a', prompt: 'p', display: 'select', options: [] },
    ]);
    expect(r.allowFreeform).toBe(false);
  });
});

describe('validateAnswers', () => {
  it('accepts a full valid set', () => {
    expect(
      validateAnswers(pending, [
        { requestId: 'r1', text: '  Ada  ' },
        { requestId: 'r2', optionId: 'green' },
      ]),
    ).toEqual({
      ok: true,
      responses: [
        { requestId: 'r1', text: 'Ada' },
        { requestId: 'r2', optionId: 'green' },
      ],
    });
  });

  const err = (answers: unknown, p = pending) => validateAnswers(p, answers);

  it('rejects an unknown request', () => {
    expect(err([{ requestId: 'nope', text: 'x' }])).toEqual({
      ok: false,
      error: 'unknown_request',
    });
  });
  it('rejects a missing answer', () => {
    expect(err([{ requestId: 'r1', text: 'x' }])).toEqual({
      ok: false,
      error: 'missing_answer',
    });
  });
  it('rejects non-array input', () => {
    for (const bad of [null, undefined, {}, 'x', 3]) {
      expect(err(bad)).toEqual({ ok: false, error: 'missing_answer' });
    }
  });
  it('rejects duplicates', () => {
    expect(
      err([
        { requestId: 'r1', text: 'a' },
        { requestId: 'r1', text: 'b' },
        { requestId: 'r2', optionId: 'red' },
      ]),
    ).toEqual({ ok: false, error: 'duplicate_answer' });
  });
  it('rejects an option not in the list', () => {
    expect(
      err([
        { requestId: 'r1', text: 'a' },
        { requestId: 'r2', optionId: 'pink' },
      ]),
    ).toEqual({ ok: false, error: 'unknown_option' });
  });
  it('rejects free text on an option-only request', () => {
    expect(
      err([
        { requestId: 'r1', text: 'a' },
        { requestId: 'r2', text: 'red' },
      ]),
    ).toEqual({ ok: false, error: 'text_not_allowed' });
  });
  it('rejects optionId plus text on an option-only request', () => {
    expect(
      err([
        { requestId: 'r1', text: 'a' },
        { requestId: 'r2', optionId: 'red', text: 'red' },
      ]),
    ).toEqual({ ok: false, error: 'text_not_allowed' });
  });
  it('rejects empty, blank, oversized or non-string answers', () => {
    for (const bad of [
      { requestId: 'r1' },
      { requestId: 'r1', text: '   ' },
      { requestId: 'r1', text: 'a'.repeat(2001) },
      { requestId: 'r1', text: 5 },
    ]) {
      expect(err([bad, { requestId: 'r2', optionId: 'red' }])).toEqual({
        ok: false,
        error: 'empty_answer',
      });
    }
  });
  it('rejects malformed entries without throwing', () => {
    expect(err([null, 'x'])).toMatchObject({ ok: false });
  });
  it('accepts a 2000 character answer', () => {
    expect(
      validateAnswers([textReq], [{ requestId: 'r1', text: 'a'.repeat(2000) }]),
    ).toMatchObject({ ok: true });
  });
});
