import Kernel from '@onkernel/sdk';
import { nanoid } from 'nanoid';
import { BrowserManager } from 'agent-browser/dist/browser.js';
import { executeCommand } from 'agent-browser/dist/actions.js';
import type { Command } from 'agent-browser/dist/types.js';
import type { ToolContext } from 'eve/tools';
import { KERNEL_TIMEOUT_SECONDS } from '@/lib/kernel/session-config';
import { cacheKey, isProfileUsable, profileNameFor } from '@/lib/kernel/session-store';

// -----------------------------------------------------------------------------
// DEVIATION FROM THE TASK BRIEF — read before touching this file.
//
// The brief's `runBrowserCommand` was meant to call `getOrCreateBrowser` from
// `lib/kernel/browser.ts` directly. That module transitively imports
// `lib/db/queries.ts`, which does `import 'server-only'`. Eve's compile/
// discovery step (v0.27.0) statically loads every `agent/tools/*.ts` file's
// whole import graph at boot to read tool schemas, and — unlike Next.js's
// webpack config, which resolves `server-only` to a no-op (`empty.js`, via the
// package's `react-server` export condition) on the server graph — Eve's
// rolldown-based bundler doesn't set that condition, so the package's default
// export (`index.js`, which unconditionally throws "This module cannot be
// imported from a Client Component module...") runs at boot and crashes `eve
// dev` before any request is served. Confirmed empirically:
//   - stub agent/tools/browser.ts (no lib/kernel/browser import): boots clean.
//   - static OR dynamic `import('@/lib/kernel/browser')`: same crash either way
//     (Eve's compile step follows dynamic imports into the same bundle graph
//     too, so deferring the import doesn't dodge it).
//   - `NODE_OPTIONS='--conditions=react-server'` (which *does* make Node's own
//     resolver pick `server-only`'s `empty.js`) fixes THAT crash, but then
//     breaks Eve's *own* internal bundling for this specific import graph —
//     requests fail with "Export 'moduleMap' is not defined in module" from
//     Eve's generated `.eve/compile/module-map.mjs`. Reproduced with the
//     diagnostic stub under the same flag: session creation succeeds fine, so
//     the corruption is specific to pulling in `lib/kernel/browser.ts`'s full
//     tree (postgres/drizzle, GCS storage, etc.) under a forced condition —
//     not something safe to ship as a workaround.
//
// So this file does NOT import `getOrCreateBrowser` from `lib/kernel/browser.ts`.
// Instead it reimplements the minimal slice of that function needed for a
// working browser tool: create-or-reuse a Kernel browser + BrowserManager,
// cached by session in this module's own process-lifetime Map. It reuses the
// two `lib/kernel/*` helper modules that are provably free of the `server-only`
// chain (`session-store.ts`'s own header says so; `session-config.ts` is pure
// constants) so the profile-naming and cache-key conventions stay identical to
// the production path.
//
// Dropped relative to `getOrCreateBrowser`: Kernel replay recording/archival to
// GCS and the `SessionMapping` DB upsert (both require `lib/db/queries.ts` /
// `lib/storage/gcs.ts`, which are exactly the modules that can't be pulled into
// this bundle). Both are already scoped to a later sub-project (replay/mapping
// is SP-C per the brief) — this file's job is only to prove the browser tool
// itself, i.e. that a `BrowserManager` (and its ref map) can be created once
// and reused across multiple Eve tool calls in the same session.
//
// Real fix for later reuse of the shared `getOrCreateBrowser` as originally
// specced: either (a) get Eve to resolve `server-only` under the `react-server`
// condition scoped to just the tool-discovery bundle (not the whole process —
// a bundler-level plugin/alias, the way Eve's own workflow-step bundler already
// does for CJS `require('server-only')`, extended to ESM `import 'server-only'`
// and to the general compile pipeline, not just workflow steps), or (b) lift
// the `server-only`-guarded DB/GCS calls in `lib/kernel/browser.ts` out of its
// module-load path (e.g. behind a lazy accessor) so importing the module for
// its browser-creation logic doesn't also drag in the replay/DB side path.
// -----------------------------------------------------------------------------

const kernel = new Kernel();

const COMMAND_TIMEOUT_MS = 120_000; // 2 minutes — Kernel commands can hang.

interface EveBrowserSession {
  browserManager: BrowserManager;
}

const sessions = new Map<string, EveBrowserSession>();
const pendingCreations = new Map<string, Promise<EveBrowserSession>>();

// Per-session serialization: Playwright's page is not concurrency-safe, so if
// Eve ever dispatches two browser commands for the same session concurrently we
// queue them. Keyed by the Eve session id; lives for the eve-dev process.
const sessionQueues = new Map<string, Promise<unknown>>();
function withSessionQueue<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
  const prev = sessionQueues.get(sessionId) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  sessionQueues.set(sessionId, next.then(() => {}, () => {}));
  return next;
}

async function ensureProfile(profileName: string): Promise<boolean> {
  try {
    await kernel.profiles.create({ name: profileName });
    return true;
  } catch (err: unknown) {
    const status = (err as { status?: number }).status;
    if (isProfileUsable(status)) return true;
    console.error('[eve-browser] Failed to ensure profile:', err);
    return false;
  }
}

/**
 * Minimal stand-in for `getOrCreateBrowser` (see the file-level comment above
 * for why this isn't the shared one). Same cache-key scheme, same profile
 * naming, no replay/DB side effects.
 */
async function getOrCreateEveBrowser(
  sessionId: string,
  userId: string,
): Promise<EveBrowserSession> {
  const key = cacheKey(userId, sessionId);

  const cached = sessions.get(key);
  if (cached) return cached;

  const pending = pendingCreations.get(key);
  if (pending) return pending;

  const createPromise = (async () => {
    try {
      const profileName = profileNameFor(sessionId);
      const hasProfile = await ensureProfile(profileName);

      const browser = (await kernel.browsers.create({
        viewport: { width: 1280, height: 800 },
        timeout_seconds: KERNEL_TIMEOUT_SECONDS,
        kiosk_mode: false,
        stealth: true,
        ...(hasProfile
          ? { profile: { name: profileName, save_changes: true } }
          : {}),
      })) as { session_id: string; cdp_ws_url: string };

      const manager = new BrowserManager();
      await manager.launch({
        id: 'launch',
        action: 'launch',
        cdpUrl: browser.cdp_ws_url,
      });

      const session: EveBrowserSession = { browserManager: manager };
      sessions.set(key, session);
      return session;
    } finally {
      pendingCreations.delete(key);
    }
  })();

  pendingCreations.set(key, createPromise);
  return createPromise;
}

// Stable browser-session identity from Eve's session context. Re-resolved on
// EVERY call (no module-held Playwright handle passed around), which is the
// durable-safe pattern from the spike's browser sketch. getOrCreateEveBrowser's
// own in-memory cache reuses the live BrowserManager within the eve-dev process.
export function browserIdentity(ctx: ToolContext): { sessionId: string; userId: string } {
  const sessionId = ctx.session.id;
  // Standalone `eve dev` has no channel auth. getOrCreateEveBrowser requires a
  // non-empty userId for cache-key isolation, so fall back to a constant.
  // Confirmed against eve's SessionAuthContext (channel/types.d.ts): `subject`
  // is the optional principal field on the authenticated caller.
  const userId = ctx.session.auth.current?.subject ?? 'eve-local';
  return { sessionId, userId };
}

export async function runBrowserCommand(
  ctx: ToolContext,
  params: Record<string, unknown>,
): Promise<{ success: boolean; data?: unknown; error?: string }> {
  const { sessionId, userId } = browserIdentity(ctx);
  return withSessionQueue(sessionId, async () => {
    const session = await getOrCreateEveBrowser(sessionId, userId);
    const command = { id: nanoid(), ...params } as Command;
    return Promise.race([
      executeCommand(command, session.browserManager),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error('Command timed out after 2 minutes')),
          COMMAND_TIMEOUT_MS,
        ),
      ),
    ]);
  });
}
