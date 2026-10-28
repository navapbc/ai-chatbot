---
description: Use when the first message starts with [extension-run] — a run in the caseworker's own browser tab through the extension. Covers the allowed browser commands, verifying values from the snapshot, asking for missing values, sensitive fields, progress reports, the one-time page re-open, iframes, and the tools you must not use.
---

## Extension Runs

This run drives the caseworker's own browser tab, and a side panel shows them what you do.

**On extension runs this skill overrides `agent/instructions.md` and the `browser-automation` and `benefits-application` skills wherever they differ.** The concrete replacements:

- Where they say to stop at `formSummary` and hand off, stop at the final review page on the site instead, and tell the caseworker to submit there themselves.
- Where they say to run `gapAnalysis`, call `formSummary`, or use `checkSubmitGate`, do not. Find gaps by walking the form (see below), ask with `ask_question`, and report with `report_progress`.
- The Submit ban in `agent/instructions.md` stays in full force.
- Where `browser-automation`, `field-patterns`, or `browser-commands` tell you to verify with `get value`, run an `eval` snippet, name an element by `#id` or a selector, use `find`, use `wait --text` or `wait --url`, group steps with `batch`, or open `tab new`, use `reload`, use `wait @ref` or take a `screenshot --full`, do not: those commands are refused on extension runs.

## Commands You May Use

Relay checks every command. Anything not listed here is refused with: "This command is not available in extension runs. Verify values from the snapshot, which masks sensitive fields." (`COMMAND_REFUSED`). Do not retry a refused command and do not look for a way around it.

Name every element by its snapshot ref (`@eN`) only. Take a `["snapshot", "-i"]` first; refs come from it. Selectors, CSS ids, xpath, and `text=` are refused.

- Look: `["snapshot", "-i"]`, `["get", "url"]`, `["get", "title"]`, `["is", "visible", "@e3"]`, `["screenshot"]`
- Act: `["click", "@e3"]`, `["fill", "@e4", "text"]`, `["type", "@e4", "text"]`, `["select", "@e5", "Option"]`, `["check", "@e6"]`, `["focus", "@e4"]`, `["scrollintoview", "@e7"]`
- Keys: `["press", "Enter"]`. Only named keys (Enter, Tab, Escape, Backspace, Delete, Space, arrows, Home, End, PageUp, PageDown), one printable character, `Control+a`, and `Meta+a`.
- Wait: `["wait", "1000"]`, `["wait", "--load", "networkidle"]`
- Move: `["scroll", "down"]`, `["frame", "@e8"]`, `["frame", "main"]`

Refused: `["eval", "x"]`, `["get", "value", "@e4"]`, `["find", "label", "Name"]`, `["batch", "x"]`, `["wait", "--text", "x"]`, `["wait", "--url", "x"]`, `["tab", "new"]`, `["click", "#id"]`, `["keyboard", "x"]`, `["dblclick", "@e3"]`, `["upload", "@e3", "x"]`, `["reload"]`, `["wait", "@e3"]`, `["screenshot", "--full"]`.

Also refused: `get text|html|attr|styles|count`, `snapshot -s`, `scroll --selector`, `drag`, `clipboard`, `screenshot --annotate`, a screenshot path, and every global flag. The caseworker does file uploads. Type into fields with `fill` or `type` on the ref, never with `batch` or key-by-key presses.

After a click that navigates, confirm it with `wait --load` then `get url`: `["wait", "--load", "networkidle"]` then `["get", "url"]`. A URL may show `[masked]` where it echoes a sensitive value; the path stays readable.

## Verifying Values

Verify a filled value from the next `snapshot`, never with `get value` or `eval`. Both are refused. Compare the field's value in the snapshot to what you typed. Sensitive fields show their value as `[masked]` once filled and empty when not filled; that is the whole check.

`[masked]` means "not for you". It can appear on field values, field names, URLs, and page text where they echo a sensitive value. Do not try to read it, copy it (no select-all or copy tricks), or probe it.

A screenshot may be refused with "The screenshot was refused because sensitive fields could not be masked." Do not retry it in a loop; continue from the snapshot.

## Asking For Missing Values

First find the gaps without `gap_analysis`: take a snapshot and walk the form page by page, comparing each field to the participant record and what the caseworker said. Then ask for every missing value in one batch, not one at a time. Make several `ask_question` calls in parallel in the same step, then wait for all the answers.

Each option has the shape `{id, label}`: choice options need both an id and a label.

- **Choice field** (a dropdown, radio group, or yes/no): pass the form's choices as `options` (each `{id, label}`) and set `allowFreeform: false` explicitly. Eve leaves `allowFreeform` off by default, and a request with options is then answered by picking one.
- **Sensitive field** (SSN, ID numbers, bank or card numbers, passwords). A field whose name shows as `[masked]` in the snapshot is a sensitive field; its label is hidden, so describe it by section and position, such as "the third field in Applicant Information": pass exactly two options, `{id: "provided", label: "I entered it myself"}` and `{id: "skip", label: "Skip"}`, with `allowFreeform: false`. The ids are exactly `provided` and `skip`. `provided` means the caseworker typed the value into the form themselves. You never type a sensitive value, never ask for it, and never repeat one.
- **Plain missing text** (a name, an address line, a phone number): ask with no options. The caseworker answers in their own words.

When the caseworker's text answer is exactly `skip`, do not fill that field: leave whatever is already in it and move on (never type `skip`). A sensitive field always gets the `provided`/`skip` options, never a free-text question.

Follow the data rules in `benefits-application`: a value that does not trace to the participant record or the caseworker is missing, not guessed.

## Progress

Call `report_progress` after each form section. Send `sections` (up to 40; each has a `name` of 1 to 80 characters, a `status` that is one of `waiting`, `filling`, `done`, or `needs_you`, a `filled` count, and an optional `total` count), an optional `current` section name (up to 120 characters), and optional `actions` (up to 5 items, each up to 120 characters). An invalid payload is silently dropped, so keep to these limits. Never put a field value in a label, name, or detail: no names, dates, numbers, or addresses of the applicant. The whole progress stream is treated as sensitive.

## Re-Opening The Page

Cross-site iframe content may not be readable until the page is opened again in the attached tab. Once per run, before any field is filled, re-open the current URL (`["open", "<current url>"]`, http or https only) and then take a snapshot. This is the only allowed use of `open` or `navigate` after the run starts. After any field is filled, never re-open: it wipes the form.

## Iframes

Switch into an iframe with `frame @<ref>`, using the ref from the snapshot. Never use a CSS selector: it does not resolve. Use `["frame", "main"]` to come back. Snapshot again after each switch.

## Tools You Must Not Use

- Do not use `gap_analysis`, `form_summary`, or `check_submit_gate`. They are for web chats.
- Do not delegate to the `form_review` subagent. Its summary tool refuses on extension runs.
- Do not hand browser work to the built-in `agent` tool or any other child session. A child runs in its own session, and its `browser` calls are refused on extension runs.
- Never click Submit, Apply, Send, or any button that finalizes the application. Stop at the final review page and tell the caseworker to submit on the site themselves.
