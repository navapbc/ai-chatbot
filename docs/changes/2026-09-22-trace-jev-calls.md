# Trace Jev calls to Braintrust

## Summary

Added `lib/jev/telemetry.ts` so every Jev (TypeSafe System One) call emits an OpenTelemetry span
and a structured stdout line. Before this, a row that came back without a `jev` annotation could
mean the flag was off, no participant record was found, or the API rate-limited — all three looked
identical in a trace.

## Changes made

- **`lib/jev/telemetry.ts`** (new) — tracer scope `labs-asp.jev`, an `ATTR` table, `withJevCallSpan`
  (one span per `askJev` request), `withJevBatchSpan` (one per tool call, carrying
  `jev.rows_attempted` / `jev.rows_annotated` / `jev.enabled`), and `annotateVerdict`. Emits a JSON
  line per batch and per failed call, following the two-output pattern in
  [`lib/observability/browser-telemetry.ts`](../../lib/observability/browser-telemetry.ts).
- **`lib/jev/client.ts`** — `askJev` takes `feature`, an optional `field`, and an optional
  `onAnswers` hook, and runs the whole call inside a span. `onAnswers` fires while the span is still
  open, which is what lets the verdict land on it; the pre-flight abort path is inside the span too,
  so a cancelled chat is distinguishable from a feature that was never reached.
- **`lib/jev/enrich.ts`** — both annotate functions wrap their work in a batch span, including the
  disabled path, and record the verdict per row. `summary-check` counts only askable rows as
  attempted, so a card full of `missing` fields does not read as a partial failure.
- **`lib/ai/tools/resolve-field-value.ts`** and **`agent/tools/resolve_field_value.ts`** — pass
  `feature` / `field` and annotate the selection.
- **`instrumentation.ts`** and **`agent/instrumentation.ts`** — replaced the single-scope
  `customFilter` with a shared `KEEP_SCOPES` set now holding the browser and Jev tracers. Both files
  need it: `filterAISpans: true` drops any unlisted scope, and the two processes export separately.
- **`instrumentation.ts`** — `OTEL_CONSOLE_SPANS=1` adds a `ConsoleSpanExporter` for local use.
- **`tests/agent/jev-telemetry.test.ts`** (new) — 5 tests over a real `BasicTracerProvider` and an
  `AsyncLocalStorage` context manager, since a no-op tracer would let these pass for the wrong
  reason.
- **Docs** — a Jev spans section, a "keeping a non-AI span" note and three local-viewing options in
  [`docs/BRAINTRUST_HOWTO.md`](../BRAINTRUST_HOWTO.md); `OTEL_CONSOLE_SPANS` in `.env.example`; a
  telemetry paragraph in `CLAUDE.md`.

## What it improves or fixes

- **A silently failing Jev is now visible.** `enrich.ts` warned that under rate limits "the feature
  would look like it was working while mostly not", and nothing could detect that state.
  `rows_annotated` below `rows_attempted` on the batch span now says so directly.
- **"Off" is distinguishable from "broken".** A disabled feature emits a span with
  `jev.enabled: false` and `jev.reason: flag_off | no_participant` rather than no span at all.
- **Verdict distributions are queryable without parsing tool output.** `jev.verdict`,
  `jev.confidence`, `jev.model` and `jev.field` arrive as `metadata.jev.*` in Braintrust — which is
  what the threshold decision was waiting on.
- **No new PII egress.** Field labels, verdicts and confidences only; the participant record, the
  entered value and the candidate list are not recorded.

## How to run / verify

```bash
pnpm exec vitest run --config vitest.config.node.mjs tests/agent/jev-telemetry.test.ts
```

Against the live services, with `TYPESAFE_API_KEY`, `JEV_FEATURES=all` and `BRAINTRUST_API_KEY` set:

```bash
OTEL_CONSOLE_SPANS=1 pnpm dev
```

Start an application from the landing page and watch for a `jev.batch.finish` line in the terminal.
In Braintrust, the spans appear under the project named by `BRAINTRUST_PARENT`:

```sql
select span_attributes.name, metadata
from project_logs
where span_attributes.name like 'jev %'
order by created desc
```

Verified on 2026-09-22 against `project_name:labs-asp-local`: a two-field `gapAnalysis` produced one
`jev gap-triage batch` span (`rows_attempted: 2`, `rows_annotated: 2`, 416 ms) and two
`jev gap-triage` children reporting `jev.model: jev-1.13.0`, verdicts `genuinely_missing` and
`present_in_record` at confidence 1.
