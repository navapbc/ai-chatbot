# Add Review And Submit

## Summary

Added plan 4c-4 to the side panel: a "Show me the unfilled field" button in Review, automatic detection of a form submission, and the confirmation number on the Confirmed screen. All of it runs in the extension worker and panel; there is no content script, no server change and no migration.

## Changes made

- `extension/src/bridge.ts`, `extension/src/typing-scripts.ts`: new worker messages.
  - `showUnfilledField(runId)` runs one read-only scan in the `relay-focus` isolated world and answers `{ok, found, remaining, sensitive?, capped?}` (counts and flags only). It highlights the first unfilled field with `Overlay.highlightNode` for about 6 s, or scrolls and focuses it when Overlay fails or the worker attached itself. Refusals are fixed strings (`registration pending`, `scan failed`, and the typing path's others).
  - `armSubmitWatch {runId, reset?}` and `disarmSubmitWatch`, plus `armed` and `submitted` on `RunStatus`. A main-frame `Page.frameRequestedNavigation` with reason `formSubmissionGet` or `formSubmissionPost`, followed within 30 s by an in-scope main-frame `Page.frameNavigated`, counts as submitted. The watch uses the agent's attach if held, else a worker attach flagged `reviewAttached` that never sets the agent-facing `attached`.
  - Confirmation read after `submitted`: a labelled pattern only, kinds `confirmation` and `other`, with drops for sensitive shapes, 9 or more digits, dates, known or registered values, emails, near-SSN text and unlisted labels. Up to 3 tries and 12 s; the token reaches only the panel through `getRunStatus`.
- `extension/panel/`: Review has the unfilled-field button and fixed status copy. `lib/run-session.ts` arms on entering Review (with reset), re-arms while `armed` is false, and stops re-arming before disarm. It auto-confirms only for kind `confirmation`; otherwise, after a 12 s wait, it asks "Did you submit the form?" (Yes and No). Confirmed shows the number ("Number shown on the page" for kind `other`). `lib/worker-api.ts`, `lib/state.ts`, `lib/copy.ts` and `lib/unfilled-field.ts` carry the types and copy.
- Tests: `tests/relay/bridge-unfilled-field.test.ts`, `bridge-submit-watch.test.ts`, `bridge-confirmation.test.ts`, panel unit tests, and new scenarios in `tests/relay/e2e/bridge.e2e.test.ts` and `panel-shell.e2e.test.ts` (including a confirmation page that echoes an SSN).
- `CLAUDE.md` and `docs/browser-extension-todo.md` updated; the spec's content-script detection is replaced by the worker watch.

## What it improves or fixes

Before, Review only said the agent had stopped, and 'I submitted it' was the only way to reach Confirmed, with no number. Now a caseworker can be pointed at the field the agent left empty, a submission is detected without a press, and the confirmation number is shown on Confirmed.

Limits: XHR and single-page-app submits are not detected (the manual button remains). A multi-step 'Next' looks like a submission, so the panel asks instead of stopping, except when the next page shows a 'Confirmation number' or code label, which stops the run irreversibly. Overlay behavior and real benefit sites are unverified. Details are under "Plan 4c-4 Follow-ups" in [the tracker](../browser-extension-todo.md).

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm exec vitest run --config vitest.config.node.mjs tests/agent tests/relay --exclude 'tests/relay/e2e/**'
pnpm extension:build && pnpm test:relay e2e/panel-shell.e2e.test.ts e2e/bridge.e2e.test.ts
```

The panel-shell e2e takes about 252 s.
