---
description: Use for any browser interaction while filling a web form — snapshotting, selectors, masked fields, native/custom dropdowns, multi-page forms, modal handling, error recovery, and the submission gate.
---

## Browser Automation

Mandatory rules for any browser action.

1. **Snapshot before interacting.** Use the refs (`@e3`) or CSS IDs (`#fieldId`) the snapshot shows. Never guess selectors. Never use `find label` when the element has an ID.
2. **No technical terms in messages.** Your audience is a caseworker. Never say refs, selectors, snapshot, DOM, CSS, eval, find label, or field IDs in your text. Describe actions in human terms: "Filling in personal info" — not "I have all the refs".
3. **Empty/minimal snapshot = modal is blocking. ALWAYS — including immediately after you just dismissed a modal.** Go straight to Modal Handling. Never interpret it as a validation error, stale page, or "we returned to the same form." Do not use `eval` to probe.

## Core Workflow

**Snapshots are your eyes. Without fresh snapshots, you are flying blind.** Every DOM change invalidates refs — re-snapshot before the next interaction or you will click the wrong element.

1. **Navigate**: `["open", "<url>"]` — already waits for load, do NOT add a separate wait
2. **Snapshot**: `["snapshot"]` — then `["snapshot", "-s", "form"]` on complex pages (Drupal, WordPress, heavy nav/sidebar)
3. **Read the refs**: Snapshots give refs like `@e3` and may show `[id="fieldId"]`. Use either for interactions — both are first-class.
4. **Interact**: `["fill", "@e3", "John"]` or `["fill", "#firstNameTxt", "John"]`
5. **Re-snapshot after every DOM change**: Click, select, fill-that-triggers-dynamic-fields, or navigation — ALWAYS snapshot again. Refs go stale after DOM changes.

**NEVER add an explicit page-load wait after clicks, fills, types, or other in-page interactions.** Only wait on the load state after navigating to a completely new URL with heavy async content (`["wait", "--load", "networkidle"]`).

## Ref Format

Snapshots return refs in this format:

```text
@e1 [button] "Submit"
@e2 [textbox name="email" id="emailTxt"] "Enter email"
@e3 [checkbox checked] "Remember me"
```

## Snapshot Modes

- `["snapshot"]` — Full page tree with labels and structure
- `["snapshot", "-i"]` — Interactive elements only (compact). Use sparingly.
- `["snapshot", "-s", "form"]` — Scoped to a container. Use on complex pages.

## Selector Rules

1. **Refs (`@e3`) or CSS IDs (`#fieldId`)** — always preferred. Use whichever the snapshot shows. CSS IDs are more stable across DOM changes.
2. **`find label`** — almost never. Only when the label is globally unique AND the element has no ID. **NEVER** use for "Yes", "No", "First Name", "Last Name", "State", "Zip Code", "Birthdate", "Phone". **NEVER** include asterisks (`*`) or colons (`:`) in the label.
3. **Tab navigation** — last resort when refs and IDs aren't working.

### Worked Example

```text
// Snapshot shows:
//   textbox  "First Name" [ref=@e3] [id="firstNameTxt"]
//   textbox  "SSN"        [ref=@e8] [id="ssnTxt"]
//   checkbox "Yes"        [ref=@e7] [id="chkBxApplyYourselfYes"]
```

```json
// Plain text — fill (ref OR id, equally valid):
["fill", "@e3", "John"]
["fill", "#firstNameTxt", "John"]

// Masked — click, select existing content, type, verify:
["click", "#ssnTxt"]
["press", "Control+a"]
["type", "#ssnTxt", "123456789"]
["get", "value", "#ssnTxt"]

// Checkbox — use the specific id to avoid ambiguity:
["check", "#chkBxApplyYourselfYes"]
```

## Masked Fields Rule

- **`fill`** = plain text only (name, address, city, email). Sets value programmatically.
- **`type`, preceded by `press Control+a` when the field isn't already empty** = masked/formatted fields (SSN, date, phone, state, zip). Simulates keystrokes so JS formatters fire. `type` never clears existing content on its own — the `Control+a` select-all first is what makes the typed keystrokes replace rather than append.
- **Respect `maxlength`**: Strip dashes/slashes/spaces. SSN → 9 digits, date → 8 digits, phone → 10 digits, state → 2 chars.
- **Always verify**: After typing into masked fields, use `["get", "value", selector]` to confirm. If wrong, click → `press Control+a` → re-type.

## Field Type Patterns

For exact argv examples for text, date, SSN, phone, state, native dropdowns, checkboxes, and radio buttons, load the sibling reference file `field-patterns.md`.

## Native `<select>` with Indexed/Coded Values

Some native `<select>` elements use numeric or coded values that don't match the visible label (e.g., BenefitsCal county picker: "Riverside" is value `"33"`, an alphabetical index). If `select` with the human-readable label fails, do NOT guess — run ONE eval to read the real option values, then retry with the correct value:

```json
["eval", "JSON.stringify(Array.from(document.querySelector('#county').options).map(o => ({value: o.value, text: o.text})))"]
["select", "#county", "33"]
```

## Custom Dropdowns

If `select` fails or has no effect (the dropdown is a custom widget like Select2, Chosen, or Drupal), load the sibling reference file `custom-dropdowns.md` for the full patterns.

## Multi-Page Forms

After clicking Next/Continue/Submit on a page, ALWAYS take a fresh snapshot. Refs from the previous page are gone — `@e1` now refers to a different element.

```json
// Page 1 — fill and advance
["snapshot", "-s", "form"]
["fill", "@e1", "..."]
["click", "@e10"]

// Page 2 — fresh snapshot required
["snapshot", "-s", "form"]
["fill", "@e1", "..."]
```

## Dynamic / Conditional Fields

When selecting an option reveals new fields, re-snapshot to discover them:

```json
["click", "@e1"]
["snapshot", "-s", "form"]
["fill", "@e5", "..."]
```

### AJAX Validation

Some fields trigger validation on blur. If you need to check for errors after filling:

```json
["fill", "@e1", "user@email.com"]
["press", "Tab"]
["snapshot", "-s", "form"]
```

## Modal Handling

Empty or minimal snapshots mean a modal is blocking the page — NOT that snapshots are broken. Modals often set `aria-hidden="true"` on the page root, hiding everything from the accessibility tree. Multiple modals can appear in sequence. Always loop until the page is clear.

**Probe budget**: If you've taken 3+ snapshots in a row that all came back minimal, you're stuck on a modal that isn't matching the standard scoped selectors. Skip ahead to *When Scoped Snapshots Also Return Empty* — do not retry the same four selectors a fourth time, do not scroll, do not click, do not reload.

### Standard Modal Workflow

1. Snapshot the page.
2. If minimal/empty content, a modal is present. Try scoped snapshots in this order:
   - `["snapshot", "-s", "[role=dialog]"]`
   - `["snapshot", "-s", ".ReactModal__Overlay"]`
   - `["snapshot", "-s", "[aria-modal=true]"]`
   - `["snapshot", "-s", ".modal"]`
3. Use refs from that snapshot to interact — native `<select>` → `select`; custom dropdown → click to open, snapshot again, click the option.
4. After dismissing, go back to step 1 — another modal may have appeared.
5. When the full page is visible again, resume normal workflow.

### Stacked Modals (BenefitsCal pattern)

After you successfully submit/dismiss a modal, your **very next action MUST be a fresh snapshot**. If that snapshot is minimal, another modal is on top — restart the Standard Modal Workflow. Do NOT:

- click anywhere on the page
- re-attempt the previous modal action
- run `eval` to "check what happened"
- assume validation failed or the click didn't register
- scroll, reload, or navigate

BenefitsCal commonly stacks county → address-confirmation → eligibility modals. Treat each "minimal snapshot after success" as a new modal until proven otherwise. If the second snapshot is also minimal after trying all four scoped selectors, jump to *When Scoped Snapshots Also Return Empty* — don't loop on the same selectors.

### When Scoped Snapshots Also Return Empty

Some modals (especially on React apps like BenefitsCal) set `aria-hidden="true"` on the root div AND lack standard ARIA attributes. Use ONE eval to discover the modal structure:

```js
["eval", "document.querySelector('[aria-modal=true], .modal, [role=dialog]')?.outerHTML?.substring(0, 2000) || 'No modal found'"]
```

If that returns nothing, try:

```js
["eval", "document.querySelector('body > div:not([aria-hidden])').outerHTML.substring(0, 2000)"]
```

Once you see the modal HTML, interact using CSS selectors (not eval):

```json
["select", "#county", "33"]
["click", "#continueBtn"]
```

### React Modals — When Select/Click Doesn't Register

React apps track form values internally. Setting `select.value` programmatically may not trigger React's state update, so the button stays disabled.

For selects — clear React's value tracker and fire change events:

```js
["eval", "var s = document.querySelector('#county'); var tracker = s._valueTracker; if (tracker) tracker.setValue(''); s.value = '33'; s.dispatchEvent(new Event('change', { bubbles: true }));"]
```

For buttons — dispatch the full mouse event sequence (not just `.click()`):

```js
["eval", "var btn = document.querySelector('button'); btn.dispatchEvent(new MouseEvent('mousedown', {bubbles:true, cancelable:true, view:window})); btn.dispatchEvent(new MouseEvent('mouseup', {bubbles:true, cancelable:true, view:window})); btn.dispatchEvent(new MouseEvent('click', {bubbles:true, cancelable:true, view:window}));"]
```

### Google Translate Bar

Government and health sites often inject a Google Translate bar that blocks clicks. Always keep the form in English — dismiss the bar if it interferes:

```js
["eval", "document.querySelector('.VIpgJd-yAWNEb-hvhgNd') && document.querySelector('.VIpgJd-yAWNEb-hvhgNd').remove()"]
```

## Error Recovery

### Field Not Found or Interaction Fails

Re-snapshot to get fresh refs. If the snapshot shows `[id="..."]` on the target field, use the CSS ID directly:

```json
["snapshot", "-s", "form"]
["fill", "#specificFieldId", "..."]
```

### Page Navigation Mid-Form

**WARNING**: `back`, `forward`, and `reload` wipe form state — all values you filled will be lost. If a page appears blank or a snapshot returns minimal content, wait and re-snapshot first. Only use `back` as a last resort, and expect to re-fill the form.

## Form Submission Protocol

**Stuck-disabled submit (Turnstile pages):** When the submit button is disabled and the page has a Cloudflare Turnstile widget, call `checkSubmitGate` once. It probes the DOM and force-enables the button so the caseworker can take control and submit. It does NOT click submit. Do not call it on pages without a Turnstile widget.

**Affirmation / expand sections are fine to complete — just don't click submit.** "Affirmation," "+ Expand," "Please read," and similar sections may need to be expanded and their checkboxes checked as part of filling the form. Do that normally. What you must NOT do is click the final submit button. If you've completed the affirmation and all required fields and the submit button is still disabled on a page with a Turnstile widget, the gate is Turnstile — call `checkSubmitGate`.

After `checkSubmitGate` runs, do NOT click submit. Proceed with `formSummary` so the caseworker can review.

## Resuming After Interruption

This section applies ONLY when there is an in-progress application from a prior turn — i.e., the caseworker says "continue" / "keep going" / "pick up where you left off", or the previous turn was clearly interrupted mid-form. On a fresh task (no prior application state), ignore this section and follow Web Search Protocol normally.

When resuming: the browser is still on the last page and mid-form. Call `["get", "url"]` and `["snapshot"]` to confirm state, then continue filling from where you stopped. NEVER call `open`, `back`, or `reload` as a recovery move — they wipe form state. NEVER restart the application from scratch unless the caseworker explicitly asks. If you can't tell where you are, stop and report to the caseworker; do not re-navigate.

## Command Format

Every command is an argv array, exactly as the agent-browser CLI takes it — one argument per array element, never quoted or shell-escaped. Flags go as their own elements: `["snapshot", "-s", "form"]`, not `["snapshot", "-s form"]`. Numbers and booleans are still passed as plain strings: `["wait", "1000"]`, `["snapshot", "-i"]` (no value needed — `-i` is a boolean flag).

## Reference Files

Load these sibling reference files on demand (see the field-type and custom-dropdown sections above for when each applies):

- `field-patterns.md` — argv examples for text, date, SSN, phone, state, dropdowns, checkboxes, radios
- `custom-dropdowns.md` — Select2 / Chosen / Drupal custom widget patterns
- `browser-commands.md` — Full command reference with all commands, flags, and options

<!-- Eve materializes these under $HOME/.agents/skills/browser-automation/ (fallback /workspace/skills/browser-automation/). At runtime, a tool/hook reaches them via ctx.getSkill('browser-automation').file('field-patterns.md') — this requires a sandbox (see Task 5). -->
