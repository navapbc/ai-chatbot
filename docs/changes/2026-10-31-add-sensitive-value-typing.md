# Add Sensitive Value Typing

## Summary

The side panel can now take a sensitive value (SSN, ID number) from the caseworker and type it into the field they focused in the tab, without the agent seeing it. The value is registered for scrubbing first and carried across token refreshes. This is plan 4c-3 (`docs/plans/2026-10-31-side-panel-sensitive-typing.md`).

## Changes made

- `extension/src/bridge.ts`: new worker message `typeSensitive(runId, value)`. It registers the value, checks that the focused element is an enabled, editable text field in a frame in scope that the redactor masks, then types into that element with one `Runtime.callFunctionOn` (`extension/src/typing-scripts.ts`, isolated `relay-focus` world), one call at a time, never retried. Refusals are fixed strings (`no active run`, `wrong run`, `not attached`, `run is held`, `value not accepted`, `no focused field`, `field not sensitive`, `field out of scope`, `type failed`).
- `extension/src/bridge.ts`: a same-run token refresh now carries the registered values to the new run (in memory; cleared on every other end).
- `lib/relay/redactor.ts`: `Redactor.masksFilledField`, the verdict for a field once it holds a value; any failure or unknown answers `false`.
- `extension/panel/lib/worker-api.ts`: `typeSensitive` with fixed result codes.
- `extension/panel/lib/sensitive-fill.ts` and `screens/sensitive-fill.tsx`: masked input and state machine. The value lives only in that controller while the card is mounted.
- `extension/panel/screens/needs-you.tsx`, `lib/question-kind.ts`: a free-text question whose prompt is sensitive gets the same card and is answered with the text `provided` or `skip`. Fixed copy in `lib/copy.ts` (`TYPE_SENSITIVE_COPY`).
- `extension/panel/lib/run-session.ts`: comments only; the panel sends `registrationComplete(runId, 0)` because the worker holds the values.
- `agent/skills/extension-run/SKILL.md`: a text answer of `provided` means the caseworker entered the value; do not ask again or try to read it.
- `CLAUDE.md` and `docs/browser-extension-todo.md`: updated; the highlighter, submit detection and confirmation number moved to plan 4c-4.
- Tests: `tests/relay/bridge-type-sensitive.test.ts`, `bridge-lifecycle.test.ts`, `panel-sensitive-fill.test.ts`, `panel-worker-api.test.ts`, `panel-needs-you.test.ts`, `panel-question-kind.test.ts`, plus e2e scenarios in `tests/relay/e2e/bridge.e2e.test.ts` and `panel-shell.e2e.test.ts`.

## What it improves or fixes

Before, a sensitive question offered only `provided` and `skip`, so the caseworker had to type the value in the page unaided, and a flagged free-text question was a dead-end card. The value was also never registered, so an echo of it could reach the model. Now the panel types it for them, the worker registers it before typing (a refusal types nothing), and later page text and screenshots are scrubbed of it. A token refresh no longer clears registered values.

Limits: 32 registered values per run, 512 characters per value; a value under 3 characters is typed but not scrubbed. A worker restart still loses the values. The caseworker focuses the field themselves; the agent's label is not used to locate it. Open items are under "Plan 4c-3 Follow-ups" in `docs/browser-extension-todo.md`.

## How to run / verify

```bash
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm exec vitest run --config vitest.config.node.mjs tests/relay/e2e/bridge.e2e.test.ts tests/relay/e2e/panel-shell.e2e.test.ts
```

The panel-shell e2e takes about 177 s.
