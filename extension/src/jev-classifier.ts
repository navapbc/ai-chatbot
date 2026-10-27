import type { FieldDescriptor } from '../../lib/relay/sensitive-fields';
import { type AppApi, DESCRIPTOR_KEYS } from './app-api';

// The redactor's classifyExtra hook for one run: asks the app's
// classify-field route (Jev) whether a field the deterministic classifier
// calls ordinary is sensitive. Jev only adds masking (P4b-e).
//
// The redactor caches every boolean the hook returns for the rest of the run,
// so a failure must never answer `false`: every failure (the request failing
// or timing out, a malformed answer, the budget running out) REJECTS, the
// redactor keeps the deterministic verdict without caching it, and the
// failure is counted. Nothing here logs, and no descriptor text outlives the
// batch it is sent in.

/** Most descriptors per request (the route refuses more). */
const MAX_PER_REQUEST = 50;
const DEFAULT_BUDGET = 200;
const DEFAULT_WINDOW_MS = 5;
// Under the redactor's default classifyTimeoutMs (2000) with the window
// added, so a hook call settles, and frees its redactor slot, on its own.
const REQUEST_TIMEOUT_MS = 1800;

export interface JevClassifier {
  /** Sensitive or not; rejects (value-free) on any failure. */
  classify: (d: FieldDescriptor) => Promise<boolean>;
  /** classify() calls that rejected, budget refusals included. */
  failures: () => number;
  /** Requests made to the app so far. */
  requests: () => number;
}

type Waiter = {
  resolve: (sensitive: boolean) => void;
  reject: (error: Error) => void;
};
type Entry = { descriptor: FieldDescriptor; waiters: Waiter[] };

const failed = () => new Error('field classification unavailable');

/** Only the descriptor's string keys, in a fixed order; never a value. */
const sanitize = (d: FieldDescriptor): FieldDescriptor => {
  const out: FieldDescriptor = {};
  const raw = d as Record<string, unknown>;
  for (const key of DESCRIPTOR_KEYS) {
    const v = raw[key];
    if (typeof v === 'string') out[key] = v;
  }
  return out;
};

export const createJevClassifier = (opts: {
  api: AppApi;
  runId: string;
  budgetPerRun?: number;
  windowMs?: number;
}): JevClassifier => {
  const { api, runId } = opts;
  const budget = opts.budgetPerRun ?? DEFAULT_BUDGET;
  const windowMs = opts.windowMs ?? DEFAULT_WINDOW_MS;
  let failures = 0;
  let requests = 0;
  // The open window: one entry per distinct descriptor, in arrival order.
  let pending = new Map<string, Entry>();
  let timer: ReturnType<typeof setTimeout> | undefined;

  const fail = (entries: Entry[]) => {
    for (const entry of entries) {
      for (const w of entry.waiters) {
        failures++;
        w.reject(failed());
      }
    }
  };

  // One request; settles within REQUEST_TIMEOUT_MS whatever the api does.
  const send = (entries: Entry[]) => {
    requests++;
    const controller = new AbortController();
    let settled = false;
    const finish = (results: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(deadline);
      if (
        !Array.isArray(results) ||
        results.length !== entries.length ||
        !results.every((r) => typeof r === 'boolean')
      ) {
        fail(entries);
        return;
      }
      entries.forEach((entry, i) => {
        for (const w of entry.waiters) w.resolve(results[i]);
      });
    };
    const deadline = setTimeout(() => {
      controller.abort();
      finish(undefined);
    }, REQUEST_TIMEOUT_MS);
    let call: Promise<boolean[]>;
    try {
      call = api.classifyFields(
        runId,
        entries.map((e) => e.descriptor),
        controller.signal,
      );
    } catch {
      finish(undefined);
      return;
    }
    Promise.resolve(call).then(finish, () => finish(undefined));
  };

  const flush = () => {
    timer = undefined;
    const entries = [...pending.values()];
    pending = new Map();
    for (let i = 0; i < entries.length; i += MAX_PER_REQUEST) {
      const chunk = entries.slice(i, i + MAX_PER_REQUEST);
      if (requests >= budget) fail(chunk);
      else send(chunk);
    }
  };

  const classify = (d: FieldDescriptor): Promise<boolean> =>
    new Promise<boolean>((resolve, reject) => {
      if (requests >= budget) {
        failures++;
        reject(failed());
        return;
      }
      const descriptor = sanitize(d);
      const key = JSON.stringify(descriptor);
      let entry = pending.get(key);
      if (!entry) {
        entry = { descriptor, waiters: [] };
        pending.set(key, entry);
      }
      entry.waiters.push({ resolve, reject });
      timer ??= setTimeout(flush, windowMs);
    });

  return {
    classify,
    failures: () => failures,
    requests: () => requests,
  };
};
