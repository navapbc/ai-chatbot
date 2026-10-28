import { type ChangeEvent, useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Textarea } from '@/components/ui/textarea';
import type {
  InputAnswer,
  PendingInputRequest,
} from '../../../lib/ai/eve/input-requests';
import { type AnswerResult, MAX_ANSWER_CHARS } from '../lib/answer-client';
import {
  AnswerAlreadyGivenError,
  AnswerRefusedError,
  answerMessageFor,
  type ErrorMessage,
  NEEDS_YOU_COPY,
  RunEndedError,
} from '../lib/copy';
import { cut } from '../lib/progress-view';
import {
  classifyQuestion,
  type QuestionKind,
  sensitiveAnswer,
} from '../lib/question-kind';

// The Needs You card: the agent's open questions, answered in one batch by
// one Send.
//
// Every prompt and option label is untrusted agent text: rendered as a React
// text node only, capped, never used in a key, an id, the URL, the title or
// storage. Typed text lives only in this component's state: never logged,
// never stored, cleared after a successful send and dropped on unmount. A
// sensitive question never gets a text input: only "I entered it myself"
// and "Skip". The card is keyed by its request ids in the parent, so a new
// question set starts fresh and the same set keeps what was typed.

export { NEEDS_YOU_COPY };
export const MAX_PROMPT = 300;
export const MAX_OPTION_LABEL = 80;

/** The id of the fixed no-numbers line under a text input. */
export const warningIdFor = (inputId: string): string => `${inputId}-warning`;

export type SendAnswers = (
  runId: string,
  answers: InputAnswer[],
) => Promise<AnswerResult>;

type Props = {
  runId: string;
  pending: PendingInputRequest[];
  sendAnswers: SendAnswers;
};

const cutText = (text: unknown, max: number): string =>
  typeof text === 'string' ? cut(text, max) : '';

type SensitiveChoice = 'provided' | 'skip';

type Draft =
  | { kind: 'text'; text: string }
  | { kind: 'skip' }
  | { kind: 'option'; index: number }
  | { kind: 'sensitive'; choice: SensitiveChoice };

type Question = {
  request: PendingInputRequest;
  kind: QuestionKind;
  prompt: string;
  labels: string[];
};

/**
 * The requests the card can show: entries with a non-empty string requestId.
 * The run state checks only that `pending` is an array, so the parent gates
 * and keys the card on this list too.
 */
export const answerableRequests = (pending: unknown): PendingInputRequest[] =>
  Array.isArray(pending)
    ? pending.filter(
        (r): r is PendingInputRequest =>
          typeof r === 'object' &&
          r !== null &&
          typeof r.requestId === 'string' &&
          r.requestId !== '',
      )
    : [];

const toQuestions = (pending: PendingInputRequest[]): Question[] =>
  answerableRequests(pending).map((request) => ({
    request,
    kind: classifyQuestion(request),
    prompt: cutText(request.prompt, MAX_PROMPT),
    labels: Array.isArray(request.options)
      ? request.options.map((o) =>
          cutText(typeof o === 'object' && o ? o.label : '', MAX_OPTION_LABEL),
        )
      : [],
  }));

/** The wire answer for one question's draft, or null when unanswered. */
const answerFor = (
  q: Question,
  draft: Draft | undefined,
): InputAnswer | null => {
  if (!draft) return null;
  const requestId = q.request.requestId;
  if (q.kind === 'text') {
    if (draft.kind === 'skip') return { requestId, text: 'skip' };
    if (draft.kind !== 'text') return null;
    const trimmed = draft.text.trim();
    if (trimmed === '' || trimmed.length > MAX_ANSWER_CHARS) return null;
    return { requestId, text: draft.text };
  }
  if (q.kind === 'choice') {
    if (draft.kind !== 'option') return null;
    const option = q.request.options[draft.index];
    return typeof option?.id === 'string' && option.id !== ''
      ? { requestId, optionId: option.id }
      : null;
  }
  if (draft.kind !== 'sensitive') return null;
  return sensitiveAnswer(q.request, draft.choice);
};

type Failure = ErrorMessage;

export const NeedsYouCard = ({ runId, pending, sendAnswers }: Props) => {
  const prefix = useId();
  const questions = toQuestions(pending);
  const [drafts, setDrafts] = useState<Map<string, Draft>>(() => new Map());
  const [sending, setSending] = useState(false);
  // 'sent' and 'waiting' keep the card disabled until the parent remounts it
  // (a new question set) or removes it (the phase moved on).
  const [done, setDone] = useState<'sent' | 'waiting' | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const mounted = useRef(true);
  const alertRef = useRef<HTMLDivElement>(null);
  const sentRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Send disables itself on press: keep keyboard focus on what replaced it.
  useEffect(() => {
    if (failure) alertRef.current?.focus();
  }, [failure]);
  useEffect(() => {
    if (done === 'sent') sentRef.current?.focus();
  }, [done]);

  const locked = sending || done !== null;
  const answers = questions.map((q) =>
    answerFor(q, drafts.get(q.request.requestId)),
  );
  const complete =
    questions.length > 0 && answers.every((a): a is InputAnswer => a !== null);

  const setDraft = (requestId: string, draft: Draft | null) => {
    setDrafts((current) => {
      const next = new Map(current);
      if (draft) next.set(requestId, draft);
      else next.delete(requestId);
      return next;
    });
  };

  const send = async (batch: InputAnswer[]) => {
    if (locked || batch.length === 0) return;
    setSending(true);
    setFailure(null);
    const result = await sendAnswers(runId, batch);
    if (!mounted.current) return;
    setSending(false);
    if (result.ok) {
      setDrafts(new Map());
      setDone('sent');
      return;
    }
    const error = result.error;
    const message = answerMessageFor(error);
    if (error instanceof AnswerRefusedError) {
      // The server does not say which answer it refused: clear every typed
      // text, keep the choices.
      setDrafts((current) => {
        const next = new Map<string, Draft>();
        for (const [id, draft] of current) {
          if (draft.kind !== 'text') next.set(id, draft);
        }
        return next;
      });
      setFailure(message);
      return;
    }
    if (
      error instanceof AnswerAlreadyGivenError ||
      error instanceof RunEndedError
    ) {
      // Nothing to retry: the next poll brings the new state.
      setDrafts(new Map());
      setDone('waiting');
      setFailure(message);
      return;
    }
    setFailure(message);
  };

  const handleSend = () => {
    if (!complete) return;
    void send(answers as InputAnswer[]);
  };

  const handleTextChange =
    (requestId: string) => (event: ChangeEvent<HTMLTextAreaElement>) => {
      setDraft(requestId, { kind: 'text', text: event.target.value });
    };

  const handleSkipText = (requestId: string, skipped: boolean) => () => {
    setDraft(
      requestId,
      skipped ? { kind: 'text', text: '' } : { kind: 'skip' },
    );
  };

  // Selects (or, pressed again, clears) the answer; Send sends the batch.
  const handleSensitive =
    (requestId: string, choice: SensitiveChoice, chosen: boolean) => () => {
      setDraft(requestId, chosen ? null : { kind: 'sensitive', choice });
    };

  const handleOption = (requestId: string) => (value: string) => {
    setDraft(requestId, { kind: 'option', index: Number(value) });
  };

  if (questions.length === 0) return null;

  return (
    <section
      aria-label={NEEDS_YOU_COPY.title}
      className="flex flex-col gap-4 rounded-xl border border-accent bg-background p-3"
    >
      <h3 className="text-sm font-semibold">{NEEDS_YOU_COPY.title}</h3>
      {questions.map((q, index) => {
        const id = `${prefix}-q${index}`;
        const requestId = q.request.requestId;
        const draft = drafts.get(requestId);
        if (q.kind === 'text') {
          const skipped = draft?.kind === 'skip';
          return (
            <div key={requestId} className="flex flex-col gap-2">
              <Label htmlFor={id} className="break-words leading-snug">
                {q.prompt}
              </Label>
              <Textarea
                id={id}
                autoComplete="off"
                spellCheck={false}
                maxLength={MAX_ANSWER_CHARS}
                aria-describedby={warningIdFor(id)}
                disabled={locked || skipped}
                value={draft?.kind === 'text' ? draft.text : ''}
                onChange={handleTextChange(requestId)}
              />
              <p
                id={warningIdFor(id)}
                className="text-xs text-muted-foreground"
              >
                {NEEDS_YOU_COPY.noNumbers}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  aria-pressed={skipped}
                  disabled={locked}
                  onClick={handleSkipText(requestId, skipped)}
                >
                  {NEEDS_YOU_COPY.skip}
                </Button>
                {skipped ? (
                  <span className="text-sm text-muted-foreground">
                    {NEEDS_YOU_COPY.skipped}
                  </span>
                ) : null}
              </div>
            </div>
          );
        }
        if (q.kind === 'sensitive') {
          const answerable =
            sensitiveAnswer(q.request, 'provided') !== null &&
            sensitiveAnswer(q.request, 'skip') !== null;
          const chosen = draft?.kind === 'sensitive' ? draft.choice : null;
          return (
            <div
              key={requestId}
              role="group"
              aria-labelledby={`${id}-prompt`}
              className="flex flex-col gap-2"
            >
              <p
                id={`${id}-prompt`}
                className="break-words text-sm font-medium"
              >
                {q.prompt}
              </p>
              {answerable ? (
                <>
                  <p className="text-sm">{NEEDS_YOU_COPY.sensitive}</p>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      variant={chosen === 'provided' ? 'default' : 'outline'}
                      aria-pressed={chosen === 'provided'}
                      disabled={locked}
                      onClick={handleSensitive(
                        requestId,
                        'provided',
                        chosen === 'provided',
                      )}
                    >
                      {NEEDS_YOU_COPY.provided}
                    </Button>
                    <Button
                      type="button"
                      variant={chosen === 'skip' ? 'default' : 'outline'}
                      aria-pressed={chosen === 'skip'}
                      disabled={locked}
                      onClick={handleSensitive(
                        requestId,
                        'skip',
                        chosen === 'skip',
                      )}
                    >
                      {NEEDS_YOU_COPY.skip}
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <p className="text-sm">{NEEDS_YOU_COPY.cannotAnswer}</p>
                  <p className="text-sm text-muted-foreground">
                    {NEEDS_YOU_COPY.shouldHaveOffered}
                  </p>
                </>
              )}
            </div>
          );
        }
        const selected = draft?.kind === 'option' ? String(draft.index) : '';
        return (
          <div
            key={requestId}
            role="group"
            aria-labelledby={`${id}-prompt`}
            className="flex flex-col gap-2"
          >
            <p id={`${id}-prompt`} className="break-words text-sm font-medium">
              {q.prompt}
            </p>
            {q.labels.length === 0 ? (
              <p className="text-sm">{NEEDS_YOU_COPY.cannotAnswer}</p>
            ) : (
              <RadioGroup
                aria-labelledby={`${id}-prompt`}
                value={selected}
                disabled={locked}
                onValueChange={handleOption(requestId)}
              >
                {q.labels.map((label, i) => (
                  // Options are positional: an agent's ids may repeat.
                  // biome-ignore lint/suspicious/noArrayIndexKey: see above
                  <div key={i} className="flex items-center gap-2">
                    <RadioGroupItem value={String(i)} id={`${id}-o${i}`} />
                    <Label htmlFor={`${id}-o${i}`} className="break-words">
                      {label}
                    </Label>
                  </div>
                ))}
              </RadioGroup>
            )}
          </div>
        );
      })}
      {failure ? (
        <div
          ref={alertRef}
          role="alert"
          tabIndex={-1}
          className="flex flex-col gap-1 text-sm focus-visible:outline-none"
        >
          <p className="font-medium">{failure.title}</p>
          <p>{failure.detail}</p>
        </div>
      ) : null}
      {done === 'sent' ? (
        <p
          ref={sentRef}
          role="status"
          aria-live="polite"
          tabIndex={-1}
          className="text-sm focus-visible:outline-none"
        >
          {NEEDS_YOU_COPY.sent}
        </p>
      ) : null}
      <Button
        type="button"
        onClick={handleSend}
        disabled={!complete || locked}
        aria-busy={sending}
      >
        {sending ? NEEDS_YOU_COPY.sending : NEEDS_YOU_COPY.send}
      </Button>
    </section>
  );
};
