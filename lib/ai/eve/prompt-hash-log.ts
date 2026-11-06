import { createHash } from 'node:crypto';
import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Prompt,
  LanguageModelV4StreamPart,
} from '@ai-sdk/provider';
import { wrapLanguageModel } from 'ai';

// A diagnostic for prompt-cache misses (a dev run on 2026-11-04 wrote 8.9M
// cache tokens, $57 on Opus 5). Anthropic caches by exact prefix, so a miss
// means some message earlier in the prompt differs from the previous step's.
// This wrapper logs, per model call, a short hash of every message and the
// tool list, the index of the first message that differs from this session's
// previous call, and the usage the provider reported. It never logs content:
// only hashes, lengths, roles, counts and token numbers, so it is safe for
// participant data. Off unless EVE_LOG_PROMPT_HASHES=1; with the flag unset
// the model is returned untouched.
//
// Reading the log (Cloud Run, filter jsonPayload.event="eve_prompt_hash"):
// a `call` line has `firstDiff` (a message index, or "append" when only
// messages were added, or "shrunk"); the `usage` line that follows has
// cacheRead and cacheWrite. A step that wrote most of its prompt with
// firstDiff a number names the message that changed; the same index across
// steps points at a rewritten message. `toolsHash` or `systemHash` changing
// between steps points at the tools or system block instead.

export const PROMPT_HASH_ENV = 'EVE_LOG_PROMPT_HASHES';
const MAX_SESSIONS = 50;

export const promptHashLoggingEnabled = (
  env: Record<string, string | undefined> = process.env,
): boolean => env[PROMPT_HASH_ENV] === '1';

const hash = (value: unknown): string =>
  createHash('sha1')
    .update(JSON.stringify(value) ?? 'undefined')
    .digest('hex')
    .slice(0, 12);

type MessageSummary = { i: number; role: string; h: string; len: number };

/** Hashes and sizes of a prompt. Content never leaves this function. */
export const describePrompt = (prompt: LanguageModelV4Prompt) => {
  const messages: MessageSummary[] = prompt.map((message, i) => ({
    i,
    role: message.role,
    h: hash(message.content),
    len: JSON.stringify(message.content)?.length ?? 0,
  }));
  const system = messages.filter((m) => m.role === 'system');
  const first = messages.find((m) => m.role !== 'system');
  return {
    // The first non-system message is the same across a session's steps, so
    // it keys the "previous call" lookup without any session id.
    session: first ? first.h : 'none',
    systemHash: hash(system.map((m) => m.h)),
    messages,
  };
};

/** Index of the first message that differs, or how the two relate. */
export const firstDifference = (
  previous: readonly MessageSummary[] | undefined,
  current: readonly MessageSummary[],
): number | 'append' | 'shrunk' | 'first' => {
  if (!previous) return 'first';
  const shared = Math.min(previous.length, current.length);
  for (let i = 0; i < shared; i++) {
    if (previous[i]?.h !== current[i]?.h) return i;
  }
  return current.length >= previous.length ? 'append' : 'shrunk';
};

type Emit = (line: string) => void;

const defaultEmit: Emit = (line) => console.log(line);

const usageSummary = (usage: unknown) => {
  const u = usage as
    | {
        inputTokens?: {
          total?: number;
          cacheRead?: number;
          cacheWrite?: number;
        };
        outputTokens?: { total?: number };
      }
    | undefined;
  return {
    input: u?.inputTokens?.total ?? null,
    cacheRead: u?.inputTokens?.cacheRead ?? null,
    cacheWrite: u?.inputTokens?.cacheWrite ?? null,
    output: u?.outputTokens?.total ?? null,
  };
};

/**
 * Wraps `model` so each call logs prompt hashes and usage. Returns `model`
 * itself when the flag is off. `emit` is injectable for tests.
 */
export const withPromptHashLogging = <M extends LanguageModelV4>(
  model: M,
  options: {
    env?: Record<string, string | undefined>;
    emit?: Emit;
  } = {},
): M => {
  if (!promptHashLoggingEnabled(options.env)) return model;
  const emit = options.emit ?? defaultEmit;
  // The previous call per session key, in this process only. A session served
  // by another instance shows as "first".
  const previous = new Map<string, readonly MessageSummary[]>();
  let step = 0;
  let current: { session: string; n: number } | undefined;

  const logCall = (params: LanguageModelV4CallOptions) => {
    const described = describePrompt(params.prompt);
    const prev = previous.get(described.session);
    step += 1;
    current = { session: described.session, n: step };
    emit(
      JSON.stringify({
        event: 'eve_prompt_hash',
        kind: 'call',
        n: step,
        model: model.modelId,
        session: described.session,
        messageCount: described.messages.length,
        systemHash: described.systemHash,
        toolsHash: hash(params.tools ?? null),
        toolCount: params.tools?.length ?? 0,
        firstDiff: firstDifference(prev, described.messages),
        messages: described.messages,
      }),
    );
    previous.delete(described.session);
    previous.set(described.session, described.messages);
    if (previous.size > MAX_SESSIONS) {
      const oldest = previous.keys().next().value;
      if (oldest !== undefined) previous.delete(oldest);
    }
  };

  const logUsage = (usage: unknown) => {
    emit(
      JSON.stringify({
        event: 'eve_prompt_hash',
        kind: 'usage',
        n: current?.n ?? null,
        session: current?.session ?? null,
        ...usageSummary(usage),
      }),
    );
  };

  return wrapLanguageModel({
    model,
    middleware: {
      specificationVersion: 'v4',
      transformParams: async ({ params }) => {
        try {
          logCall(params);
        } catch {
          // A diagnostic never breaks a model call.
        }
        return params;
      },
      wrapGenerate: async ({ doGenerate }) => {
        const result = await doGenerate();
        try {
          logUsage(result.usage);
        } catch {}
        return result;
      },
      wrapStream: async ({ doStream }) => {
        const { stream, ...rest } = await doStream();
        const logged = stream.pipeThrough(
          new TransformStream<LanguageModelV4StreamPart>({
            transform(part, controller) {
              if (part.type === 'finish') {
                try {
                  logUsage(part.usage);
                } catch {}
              }
              controller.enqueue(part);
            },
          }),
        );
        return { stream: logged, ...rest };
      },
    },
  }) as M;
};
