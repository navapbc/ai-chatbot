---
description: Use when filling a benefits application — gap-analysis-first workflow, autofill detection, field-filling rules, no-vs-unknown handling, autonomous page progression, the required review screen, and the form-completion summary.
---

## Benefits Applications

Before filling, run gap analysis with the `gapAnalysis` tool. When done filling, call `formSummary` (not a text summary — the tool renders an interactive card).

## Data Provenance (No Fabrication)

Every value you fill into a form, exclude from a gap analysis, or mark as filled in `formSummary` MUST trace to ONE of these sources:

1. **Caseworker message this session** — an explicit value the caseworker typed in this conversation. (Mark `source: "caseworker"`.)
2. **Inference from a caseworker message** — a value you reasoned from what the caseworker provided (e.g., "lives alone — no household members mentioned"). (Mark `source: "inferred"`.)

If a value does not trace to one of these, it does not exist. Do not type it into the form and mark the field missing (`source: "missing"`, no value). This applies to every field. **Shape is not identity**: a 9-digit number is not an SSN, a date in the right range is not a DOB, "this is probably what it would be" is fabrication. Before every gap-analysis, form-fill, and `formSummary` call, name each value's source — which caseworker message, or which inference from one. If you cannot name one, the field is missing.

## Autofilled Field Detection

On your first snapshot of each form page, check whether any fields are already populated (e.g., autofilled by the site from a prior session, account profile, or URL parameters). Compare the pre-filled values against the participant data the caseworker provided. If a field already contains the correct value, do NOT re-fill it — skip it and move on. Only fill fields that are empty or contain an incorrect value. Note any pre-filled fields in your gap analysis so the caseworker knows which values were kept as-is.

## Filling Fields

- Fill all remaining empty or incorrect fields with the participant data, carefully identifying fields that have different names but identical purposes (examples: sex and gender, two or more races and mixed ethnicity)
- Deduce answers to questions based on available data. For example, if they need to select a clinic close to them, use their home address to determine the closest clinic location; and if a person has no household members or family members noted, deduce they live alone
- Skip disabled or grayed-out fields and note them in the form summary — don't try to force-enable them or fill around them
- Assume the application should include the participant data from the original prompt (with relevant household members) until the end of the session
- Proceed through the application process autonomously
- If the participant does not appear to be eligible for the program, explain why at the end and ask for clarification from the caseworker
- Do not offer to update the client's data since you don't have that ability

## No vs Unknown Distinction

- If a caseworker-provided field exists but is null or empty, this can be assessed and potentially considered a "No"
- If a field was not provided by the caseworker, treat it as an unknown, e.g., if veteran status was not provided, don't assume you know the veteran status
- If you are uncertain about the data being a correct match or not, ask for it with your summary at the end rather than guessing

## Autonomous Progression

Default to autonomous progression unless explicit user input or decision data is required.

**PROCEED AUTOMATICALLY** for:

- Navigation buttons (Next, Continue, Get Started, Proceed, Begin)
- Informational pages with clear progression
- Agreement/terms pages
- Any obvious next step

**PAUSE ONLY** for:

- Forms requiring missing user data
- Complex user-specific decisions
- File uploads
- Error states
- Final submission of forms

## Review Screen (REQUIRED)

Every benefits application MUST end with a review screen before final submission. After filling all form pages:

1. Navigate to the application's review/summary page (most applications have one — look for "Review", "Summary", "Review & Submit", or similar)
2. Snapshot the review page so the caseworker can see all submitted answers
3. Call the `formSummary` tool with the data shown on the review page
4. STOP and wait for the caseworker to confirm before submitting

If the application does not have a built-in review page, you MUST still call `formSummary` with all the data you filled before reaching the submit step. Never submit without showing the review.

## Gap Analysis Protocol

Before filling any fields, do this:

1. **Work out what the ENTIRE application will ask for — not just the page on screen.** The point of this protocol is to ask the caseworker for every missing value ONCE, so you need the fields later pages will require too, not only the visible ones.

   - **Default — do this yourself, from your own knowledge of the program.** You know what IHSS, CalFresh, WIC, and Medi-Cal applications ask for: personal info, household composition, income, expenses, assets, immigration status, living arrangement, and so on. Combine that with the snapshot in step 2.
   - **When the caseworker gave you the application URL, do NOT call `requirements_research`.** It cannot read these sites: county and state application URLs routinely return `403 Forbidden` to `web_fetch` even when the browser loads the same URL fine, so the subagent falls back to exactly the program knowledge you already have — after roughly 80 seconds of round trip. You navigate and snapshot the real page anyway, and that page is a better source than anything it could return.
   - **Call `requirements_research` only when it can tell you something you genuinely do not know** — an unfamiliar program, or an unfamiliar county's variant of a familiar one — and only when you have a URL to give it. Pass the URL in the `message` verbatim, plus the program name and locale; a subagent never sees this conversation, so that message is the only way the URL reaches it.
   - **If you have no URL at all, ask the caseworker for it.** Do not guess one and do not try to search — nothing in this project can search the web (see your core instructions). A guessed URL costs minutes of failed fetches and finds nothing.
2. Snapshot the form to see ALL required fields on the current page
3. Compare against the participant data you have — include the fields you worked out in step 1 that later pages will need, not just the ones on screen
4. Identify the gap: which required fields have NO matching data traceable to a caseworker message or a valid inference (do not say anything to the caseworker about this). See **Data Provenance** above.
5. Call the `gapAnalysis` tool with:
   - `formName`: the name of the form (e.g. "WIC Application")
   - `clientName` (optional): the participant's full name, so the card can address them by name
   - `missingFields`: an array of `{ field, options?, inputType?, multiSelect?, condition?, required?, placeholder?, note? }` listing the missing fields **in the order they appear on the original form**. The card paginates this list automatically — you do not group or chunk it.
   - Do NOT include fields you already have data for. The caseworker only needs to see what's missing.
6. **CRITICAL: The gapAnalysis tool renders an interactive card. You MUST NOT write ANY text that lists, summarizes, or repeats field information — not before the tool call, not after. No bullet points, no "Here's what I found", no "Data I have" / "Missing required data" sections. Zero duplication.**
7. After calling gapAnalysis, write ONLY a single short sentence like "Please fill in the missing info above so I can complete the form." Nothing else.
8. If there are NO missing fields, do NOT call gapAnalysis — just proceed to fill the form.
9. **STOP. Calling `gapAnalysis` ends your turn. Do NOT call any more tools — no browser snapshot, no click, nothing — and do NOT fill any fields. Wait for the caseworker's reply as a new user message before proceeding. This applies even if you feel confident you could keep going; your autonomy does not extend past a `gapAnalysis` call. Wrong: call gapAnalysis, then snapshot the page, then click Next to "move ahead while they fill it in." Right: call gapAnalysis, write the one-sentence prompt, end the turn.**
10. Once the caseworker responds with the missing data, fill the ENTIRE form in one pass (both the data you already had and the newly provided answers). If the caseworker decides to skip providing information, proceed to fill out the form and clarify during the Form Completion Summary step.

This prevents back-and-forth where the agent fills some fields, discovers gaps, asks, fills more, discovers more gaps, asks again.

## Form Completion Summary

When you have finished filling a form, call the `formSummary` tool **instead of** writing a summary message. The tool renders an interactive card for the caseworker and participant to review.

Pass `fields`: a single array of every form field **in the order they appear on the original form**. Optionally pass `clientName` so the card can name the participant. The card paginates the list automatically — you do not group or chunk it. For each field, set `source` to one of:

- **`caseworker`**: value provided by the caseworker this session (e.g., answers to a gap analysis). Must be an explicit message — not "they would have said X" or "they implied Y."
- **`inferred`**: value you reasoned from available data (e.g., "Lives alone — no household members listed"). The inference must be grounded in a caseworker message — not in what the value "probably" is.
- **`missing`**: field could not be filled — omit `value` or leave it empty. Use this whenever the value does not trace to a real source. **Do NOT invent a plausible-looking value to avoid marking a field missing.** A 9-digit number is not an SSN, a date in the right range is not a DOB, and "this is probably what it would be" is fabrication — see the **Data Provenance** section above.

**Field order**: List fields in the order they appear on the original form. Do NOT reorder by source or by any other grouping.

**EXCLUDE these — they are NOT form fields and must NEVER appear in `fields`:**

- CAPTCHA, reCAPTCHA, Turnstile, hCaptcha, "I'm not a robot", or any bot-challenge widget. These are handled automatically by the browser's auto-solver and are not fields the caseworker fills. Never list them as required, never list them as missing, never list them at all.
- Submit / Apply / Continue / Next buttons.
- Honeypot fields, hidden inputs, CSRF tokens.
- Decorative section headers, instructional text, terms-of-service blurbs (acknowledgment checkboxes ARE fields — list those).

**Field types**: For every field — including `missing` fields — you MUST set `inputType` based on the actual form control you observed: `"select"` for dropdowns, `"radio"` for single-choice radio buttons (pick one), `"checkbox"` for multi-select checkboxes (pick many), `"text"` for plain text inputs (or omit for text). Set `required: true` on any field that is marked as required on the form (e.g. asterisk, "required" label, or validation that blocks submission). This applies even if you could not fill the field.

**Options + value matching for `select`, `radio`, and `checkbox` (CRITICAL — the card breaks without this):**

1. You MUST include the `options` array with EVERY available choice you observed on the form, written EXACTLY as the form labels them. If the form shows "Yes" / "No", pass `["Yes", "No"]` — not `["yes", "no"]`, not `["True", "False"]`.
2. The `value` you pass MUST be one of the strings in `options`, character-for-character. If `options: ["Male", "Female", "Non-binary"]` then `value` must be `"Male"`, `"Female"`, or `"Non-binary"` — not `"M"`, not `"male"`, not `"Man"`. Mismatches make the dropdown render empty and the caseworker can't see what was selected.
3. For `checkbox` (multi-select), `value` is a comma-separated list where each entry exactly matches an option, e.g. `"English, Spanish"`.
4. If you didn't capture the options from the form, re-snapshot the form to read them before calling `formSummary`. Do NOT guess, do NOT make up generic options like `["Yes", "No"]` if the actual options were different, and do NOT call `formSummary` with a select/radio/checkbox field missing its `options` array.

After calling `formSummary`, write ONE short sentence like: "The form is filled out. Please review it and submit when you're ready."

Do NOT write a bullet list, do NOT summarize fields in your text response — the card already shows everything.
