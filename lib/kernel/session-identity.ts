import type { ToolContext } from 'eve/tools';

// Pure and dependency-free (type import only) so both the Kernel and relay
// browser paths can share it without pulling in the Kernel SDK client, which
// `lib/kernel/eve-browser.ts` constructs at module load.

// Stable browser-session identity from Eve's session context. Re-resolved on
// EVERY call (no module-held browser handle passed around), which is the
// durable-safe pattern from the spike's browser sketch.
export function browserIdentity(ctx: ToolContext): {
  sessionId: string;
  userId: string;
} {
  const sessionId = ctx.session.id;
  // Prefer `initiator` over `current`: eve pins `initiator` to whoever created
  // the session, while `current` tracks the caller of the most recent request
  // (see eve's SessionAuth in context/keys.d.ts). A browser belongs to the
  // conversation, so it must be keyed on the session-stable principal — keying
  // on `current` changed this id on the first follow-up turn, which renamed the
  // agent-browser daemon and silently orphaned the browser and its `@eN` refs
  // mid-form. `agent/agent.ts` resolves eveModel with the same precedence.
  //
  // Resolve one principal and read both fields off it, so we never mix
  // initiator's subject with current's principalId. `subject` is optional on
  // SessionAuthContext but `principalId` is required, so fall through to it
  // rather than collapsing a real principal to the constant below.
  // Standalone `eve dev` has no channel auth at all, and the browser caches
  // require a non-empty userId for isolation — hence the constant.
  const principal = ctx.session.auth.initiator ?? ctx.session.auth.current;
  const userId = principal?.subject ?? principal?.principalId ?? 'eve-local';
  return { sessionId, userId };
}
