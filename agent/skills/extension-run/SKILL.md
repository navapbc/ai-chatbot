---
description: Use when the first message starts with [extension-run] — a run in the caseworker's own browser tab through the extension. Covers asking for missing values, sensitive fields, progress reports, the one-time page re-open, iframes, and the tools you must not use.
---

## Extension Runs

This run drives the caseworker's own browser tab, and a side panel shows them what you do. The rules below add to the `browser-automation` and `benefits-application` skills. Where they differ, follow this skill.

## Asking For Missing Values

Ask for every known missing value in one batch, not one at a time. Make several `ask_question` calls in parallel in the same step, then wait for all the answers.

- **Choice field** (a dropdown, radio group, or yes/no): pass the form's choices as `options` and set `allowFreeform: false` explicitly. Eve leaves `allowFreeform` off by default, and a request with options is then answered by picking one.
- **Sensitive field** (SSN, ID numbers, bank or card numbers, passwords): pass exactly two options, `provided` and `skip`, with `allowFreeform: false`. `provided` means the caseworker typed the value into the form themselves. Never ask for the value, never repeat it, and never type it yourself.
- **Plain missing text** (a name, an address line, a phone number): ask with no options. The caseworker answers in their own words.

Follow the data rules in `benefits-application`: a value that does not trace to the participant record or the caseworker is missing, not guessed.

## Progress

Call `report_progress` after each form section. Send section names, statuses, filled and total counts, the current section, and short recent actions. Never put a field value in a label, name, or detail: no names, dates, numbers, or addresses of the applicant. The whole progress stream is treated as sensitive.

## Re-Opening The Page

Cross-site iframe content may not be readable until the page is opened again in the attached tab. Once per run, before any field is filled, re-open the current URL (`["open", "<current url>"]`) and then take a snapshot. This is the only allowed use of `open` or `navigate` after the first page load. After any field is filled, never re-open: it wipes the form.

## Iframes

Switch into an iframe with `frame @<ref>`, using the ref from the snapshot. Never use a CSS selector: it does not resolve. Use `frame main` to come back. Snapshot again after each switch.

## Tools You Must Not Use

- Do not use `gap_analysis`, `form_summary`, or `check_submit_gate`. They are for web chats.
- Do not delegate to the `form_review` subagent. Its summary tool refuses on extension runs.
- Never click Submit, Apply, Send, or any button that finalizes the application. Stop at the final review page and tell the caseworker to submit on the site themselves.
