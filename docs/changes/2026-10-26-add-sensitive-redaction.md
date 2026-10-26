# Add sensitive-value redaction to extension runs

## Summary

Extension runs now keep sensitive values (SSN, bank and card numbers, passwords, ID numbers) out of everything the model, the relay and the traces carry. The worker masks replies, events and screenshots, the tool layer allowlists commands, the route strips sensitive record fields, and Eve spans are scrubbed.

## Changes made

Roughly 46 files, 43 commits on top of `browser-extension` (`git diff browser-extension...HEAD --stat`).

- Classifier: `lib/relay/sensitive-fields.ts` (pure; label, name, id, autocomplete, placeholder) with `evals/datasets/sensitive-fields.json` and a Jev question and eval (`lib/jev/questions.ts`, `evals/jev-questions.eval.ts`) for regression scoring.
- Redaction: `lib/relay/redactor.ts` and `extension/src/redaction.ts` mask sensitive fields in accessibility trees, `DOM.describeNode`, URLs and `Runtime.*` results, scrub registered known values in every encoding, and mask value-shaped data. `extension/src/bridge.ts` routes every reply and event through them and exposes `addKnownValue`.
- Screenshots: `lib/relay/screenshot-mask.ts` masks sensitive fields in pixels under a per-tab lock, and refuses when it cannot or when a registered value is on the page.
- Frames: `lib/relay/frames.ts` scopes same-site cross-origin iframes per session; child session URLs follow navigation events.
- Events and Page domain: `lib/relay/policy.ts` adds a 24-method event allowlist; the bridge enables the Page domain itself and denies `Page.disable`.
- Commands: `lib/relay/command-policy.ts`, called first by `lib/relay/eve-relay-browser.ts`, allows only listed commands that name elements by `@eN`. `eval`, value `get`, `find`, `batch`, selectors and global flags are refused. `agent/skills/extension-run/SKILL.md` tells the agent how to verify from masked snapshots.
- Records: `lib/data/sensitive-record.ts` and `app/(chat)/api/eve-chat/route.ts` strip sensitive fields from the opening message and relay follow-ups, refuse messages over 256 KB (`message_too_large`) and refuse prose that pairs a sensitive label with a value (`sensitive_value_in_message`). Free-text answers to `ask_question` get the same size cap and guard, checked with the question's prompt in front, before the token claim and with pending left as it was. `lib/jev/participant.ts` and `lib/jev/flatten.ts` skip withheld fields.
- Traces: `lib/observability/scrubbing-exporter.ts` and `agent/instrumentation.ts` export only allowlisted span attributes when `RELAY_RUNS_ENABLED=1`.
- Tests: unit tests under `tests/relay/` and `tests/agent/`, and `tests/relay/e2e/redaction.e2e.test.ts` with fixtures and harness changes (`tests/relay/fixtures/`, `tests/relay/helpers/harness.ts`).
- Docs: `CLAUDE.md`, `agent/README.md`, `docs/browser-extension-todo.md`.

## What it improves or fixes

Before, the extension sent accessibility trees, `DOM.describeNode` replies, URLs, screenshots and script results up the relay unredacted, and the agent could read any field with `eval` or `get value`. The participant record, including `calworks_id`, went into the opening message, and traces carried tool arguments and results. Same-site cross-origin iframes shared the tab session without a frame check, and leave-origin detection depended on the agent enabling the Page domain.

After, a sensitive field shows as `[masked]` (empty stays empty), so the agent can tell filled from empty but cannot read the value. The end-to-end tests drive the real `agent-browser` through the relay against a fixture with sensitive fields and check that no sensitive string, token or CDP URL appears in any tool result or relay frame, including after a GET-submitted form puts a value in the URL.

Accepted residuals and the rulings behind them are in [the tracker](../browser-extension-todo.md) (see "Known limits (accepted for plan 3)" and "Plan 3 rulings"). Notable ones: values echoed in ordinary page text are not masked unless registered, nothing registers values until the plan 4 panel, and the chat prose guard is best-effort. Running agent-browser end to end showed that `wait @ref` times out and that the bridge denies `reload` and `screenshot --full`, so the command policy refuses all three.

## How to run / verify

```bash
export PATH=/Users/kaylynvannorstrand/.nvm/versions/node/v24.20.0/bin:$PATH
pnpm extension:build && pnpm test:relay
pnpm test:relay e2e/redaction
pnpm exec vitest run --config vitest.config.node.mjs tests/agent
```

`pnpm test:relay` includes the Chromium e2e files. The Jev classifier eval (`pnpm eval:jev-questions` or `pnpm eval` with `.env.local`) was not run as part of this change; run it before relying on the classifier scores, since the judge now needs all three Jev features enabled.

`pnpm exec vitest run --project browser` still has 3 failures (`form-summary-card.test.tsx` twice, `gap-analysis-card.test.tsx` once). They fail on a clean `browser-extension` checkout and are not part of this change.
