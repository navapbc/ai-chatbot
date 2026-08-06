# Universal Form-Completion Protocol

Site-agnostic discipline for completing a form on ANY website. Load at the start of a new
form task, before the first fill. Every rule here maps to a real failure observed in the
field — none of it is theoretical.

## Phase 0 — Reach the actual form

The URL you are given is frequently a landing page, not the form.

- `["snapshot", "-i", "-u"]` — the `-u` flag prints link hrefs. Follow apply / continue /
  start links; expect 1–3 interstitial hops before the real form.
- **Signal test before filling anything**: a nonzero input count does not mean a form is
  present. Check id/name prefixes — Google Translate (`goog-gt-*`), chat widgets, search
  boxes, and cookie banners all contribute inputs. A page whose only inputs belong to
  widgets has NO form.

## Phase 1 — Map fields and resolve ALL gaps before filling

1. Enumerate fields. Selector preference: **`#id` > `[name=…]` > `@eN` refs.** Snapshot refs
   go stale after DOM mutation and cannot distinguish repeated "Yes/No" pairs (section
   headings often render AFTER the fields they label).
2. Identify required fields (`*` in labels, `required`/`aria-required`).
3. Run the gap analysis against participant data, then ask the caseworker for **all**
   missing values and path decisions in ONE message — never field-by-field mid-fill.
   Path decisions that change the fill: "applying for self or on behalf?", "mailing address
   same as residential?", account type.
4. Never invent required values. Never substitute look-alike identifiers (a case
   number is not an SSN).

## Phase 2 — Fill in gate order, top to bottom

- **Master gates first.** Early questions reshape the form below them. A field hidden
  behind an unanswered gate accepts commands with a success message while doing nothing.
- **Gate polarity is not guessable.** "Same as above? → Yes" HIDES the dependent block —
  that is correct behavior, not a failure. Read the question text before choosing.
- Use `["check", "@e1"]` / `["uncheck", "@e1"]` for checkboxes — they are idempotent
  setters. `click` TOGGLES: two clicks restore the original state, and clicking "No" then
  "Yes" in a pair can leave both unchecked (Yes/No pairs are often independent checkboxes,
  not radios). After writing, read back the whole pair.
- Selects: match option text EXACTLY — enumerate options first. Data may say
  "Spanish (Mexico)" where the form only offers "Spanish". `["get", "value", …]` on a
  select returns the option INDEX, not its text.

## Phase 3 — Verify EVERY write; success output is meaningless

The most dangerous failure class is the **silent success**: the command reports done, the
field holds nothing. Read back every field after writing (`get value`, `is checked`).

Diagnose a failed/empty field in this order:

1. `["is", "enabled", "@e1"]` false → gated by a checkbox/select; set the gate, refill.
2. `["is", "visible", "@e1"]` false → an ancestor is hidden; answer the upstream question.
   (If the block is *meant* to stay hidden — e.g. mailing same as residential — empty is
   correct; fill nothing.)
3. Value shows `__/__/____` or `(___) ___-____` → masked input; use the escalation ladder
   in `field-patterns.md` (ending in per-character `key` presses).
4. `["get", "attr", "@e1", "maxlength"]` shorter than your value → rejected or truncated
   wholesale: 2-char state wants `CA` not `California`; 8-char date wants `MMDDYYYY`;
   units/apartments append to the street line.

Never re-trust a stale snapshot after the DOM changes — expanders can still show
"+ Expand" in old refs after successfully expanding. Verify state with `get text` /
`is visible` on a fresh selector.

## Phase 4 — Exceptions

When the taxonomy above doesn't explain a failure, slow down and investigate — but test a
hypothesized cause before acting on it. Observed misdiagnoses to avoid repeating: blaming
a bot-check for a disabled submit (the real gate was an unexpanded affirmation section);
treating a correctly-hidden block as a bug (wrong polarity assumption).

## Phase 5 — Submit is a human gate

- If submit is disabled, diagnose THIS site's enablement condition rather than assuming
  one. Candidates to check: missing required fields, unchecked consent boxes, unexpanded
  disclosure/affirmation sections, unresolved bot challenges. Test candidates one at a
  time and confirm with an enabled-state check which one flips it.
- Bot challenges (Turnstile, reCAPTCHA): use `checkSubmitGate` for stuck-disabled
  submits — never try to defeat a challenge. Verify the challenge is actually the gate
  before blaming it; if it is, stop and tell the caseworker.
- Before submitting, present the caseworker a field-by-field diff of verified values vs
  source data and get explicit approval. Never submit autonomously on applications with
  legal effect. Never submit test data at all.
