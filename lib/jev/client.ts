// Shared entry point for every Jev (TypeSafe System One) call in the app.
//
// Deliberately free of `server-only` and of anything that transitively pulls
// it in (notably `lib/db/queries.ts`): the Eve runtime bundles its own tools
// and cannot import such a module, so keeping this dependency-free is what
// lets agent/tools/* reuse it instead of growing a second copy. See the
// lib/kernel/browser.ts vs lib/kernel/eve-browser.ts split for the duplication
// this avoids.
//
// Jev is called from code with questions written in code — not exposed as a
// tool the model composes at runtime. Fixed questions can be versioned,
// reviewed, and regression-scored; model-authored ones cannot.

import {
  TypeSafeClient,
  type Questions,
  type SystemOneResult,
} from '@typesafe-ai/sdk';

/** Feature keys accepted in `JEV_FEATURES`. */
export type JevFeature = 'gap-triage' | 'summary-check' | 'field-value';

const ALL_FEATURES: readonly JevFeature[] = [
  'gap-triage',
  'summary-check',
  'field-value',
];

// One comma-separated env var rather than a boolean per feature, so enabling a
// subset is a single terraform/`.env.local` edit:
//
//   JEV_FEATURES=gap-triage,summary-check
//
// `JEV_ONLINE_SCORING` (evals/online/rules.ts) is a separate boolean on
// purpose — it gates a CLI that pushes config to Braintrust, runs in a
// different process, and never reads this.
//
// `lib/feature-flags.ts` is not used for any of these: it resolves overrides
// from `localStorage`, and `getFlagOverride` returns null when there is no
// `window`, so on the server only `defaultValue` would ever apply. Every Jev
// call site here is server-side.
const parseFeatures = (raw: string | undefined): Set<JevFeature> => {
  if (!raw) return new Set();
  if (raw.trim() === 'all') return new Set(ALL_FEATURES);
  const named = raw
    .split(',')
    .map((s) => s.trim())
    .filter((s): s is JevFeature =>
      (ALL_FEATURES as readonly string[]).includes(s),
    );
  return new Set(named);
};

/**
 * Whether `feature` should run. False when the flag omits it OR when
 * `TYPESAFE_API_KEY` is unset — a missing key is treated as "off" rather than
 * an error so a misconfigured deploy degrades to today's behavior instead of
 * failing a caseworker's form run.
 */
export const isJevEnabled = (feature: JevFeature): boolean =>
  Boolean(process.env.TYPESAFE_API_KEY) &&
  parseFeatures(process.env.JEV_FEATURES).has(feature);

// Constructed lazily: the constructor throws when TYPESAFE_API_KEY is absent,
// and importing this module must stay safe in tests and in environments that
// never enable a Jev feature.
let client: TypeSafeClient | undefined;
const getClient = (): TypeSafeClient => {
  if (!client) client = new TypeSafeClient();
  return client;
};

/** Reset between tests; no effect in production paths. */
export const resetJevClientForTests = (): void => {
  client = undefined;
};

export type JevOutcome<Q extends Questions> =
  | { ok: true; answers: SystemOneResult<Q>['answers']; model: string }
  | { ok: false; reason: string };

/**
 * Ask Jev a fixed set of questions about one piece of state.
 *
 * Never throws. Jev here is an enrichment on top of paths that already work
 * without it, so any failure — bad key, rate limit, timeout, caller abort —
 * resolves to `{ ok: false }` and the caller falls back to its original
 * behavior. Callers must handle that branch; nothing in the app should block
 * a form run on Jev being reachable.
 *
 * Questions run in parallel inside a single request, so ask everything that
 * shares this `state` in one call rather than awaiting several.
 */
export async function askJev<const Q extends Questions>(args: {
  state: unknown;
  questions: Q;
  /** The tool's own `abortSignal`, so a cancelled chat cancels this too. */
  signal?: AbortSignal;
  /** Per-attempt timeout. Below the SDK's 10s default: these run inline in a turn. */
  timeoutMs?: number;
}): Promise<JevOutcome<Q>> {
  const { state, questions, signal, timeoutMs = 5000 } = args;
  if (signal?.aborted) return { ok: false, reason: 'aborted' };
  try {
    const result = await getClient().systemOne(
      { state: state as never, questions },
      { signal, timeout: timeoutMs },
    );
    return { ok: true, answers: result.answers, model: result.model };
  } catch (error: unknown) {
    return {
      ok: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}
