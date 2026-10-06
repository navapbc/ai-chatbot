---
name: form-completion
description: Complete, test, or debug a web form with agent-browser — application, benefits, or government forms, multi-step apply flows, or any site form filling. Use when asked to fill out a form, apply on a website, drive agent-browser against a page, or when a form field silently fails to fill (fill reports done but the value shows __/__/____ or stays empty).
---

# Form Completion With agent-browser

This skill gives a procedure to fill a web form on a website. Do the phases in
sequence. Each rule comes from a failure that occurred in a real session.

The most dangerous failure is the silent success. The tool shows `✓ Done`, but the
field stays empty. Phase 4 finds these failures. Do not skip Phase 4.

Find the binary in this sequence: `$AGENT_BROWSER_BIN`, then
`./node_modules/.bin/agent-browser`, then `agent-browser` on the PATH. You start and
control the browser — do not ask the user to do it. With no `--cdp` flag the CLI
starts its own browser; add `--headed` when the user wants to watch. To attach to the
user's own Chrome instead, start Chrome with `--remote-debugging-port=9222` and a
dedicated `--user-data-dir`, then pass `--cdp http://localhost:9222`. A remote
provider (Kernel, Browserbase) gives a CDP WebSocket URL for `--cdp`.

**When one run has two or more applications**, read `references/multi-application.md`
first. The main session becomes the orchestrator: it does the intake for all the
applications, then starts one fill agent for each application in parallel. For an
application with no playbook, a background scout agent surveys the site — the
orchestrator stays with the user and does not survey a site itself. Each fill agent
gets its own browser. Do not give one page to two writers.

## Model for each dispatched agent

**This applies whenever you start an agent — including a single-application run.** A cold
start dispatches a scribe, and a scout if the form has to be found, so most runs dispatch
something. Previously this guidance lived in `references/multi-application.md`, which
`SKILL.md` only sent you to for two or more applications; single-form runs never read it and
so never passed a `model` parameter.

The `Agent` tool takes a `model` parameter. **Pass it every time.** Without it the agent
inherits the orchestrator model, which costs roughly ten times more for each input token.

| Agent | Model | Reason |
|---|---|---|
| Scout | `sonnet` | It must find the real form behind menus and interstitial pages. |
| Scribe | `sonnet` | It writes the knowledge files that later runs depend on. |
| Fill agent, no playbook (cold start) | `sonnet` | Discovery needs judgment: label mapping, gate polarity. |
| Fill agent, playbook present (warm path) | `sonnet` — see below | Was `haiku`. Changed on measured evidence. |

**Why the warm fill agent is no longer `haiku` by default.** The reasoning for `haiku` was
that the fill agent reads the large DOM outputs, so a cheaper model controls most of the run
cost. Measured on 2026-09-02, that did not hold, for a reason the reasoning did not
anticipate: **the volume of reading is not fixed across models.** A `haiku` fill agent on a
form whose playbook documented every field took **135 browser commands, 56 of them whole-page
`snapshot` calls, and re-opened the form 6 times** — each re-open discarding the fills. It
never returned BLOCKED, and it reported a site fact that was false. The run cost more than
doing the whole job on `sonnet` in one session, and the cost was a symptom of the thrash, not
a pricing effect.

`haiku` is still the right choice for a fill agent whose whole sequence is a script — see
*Prefer a site script* — because a script removes the judgment the small model was failing
at. Until a site has one, use `sonnet` for the fill agent and record the exception.

## Phase 0 — Find the Playbook

Look in the `playbooks/` directory of this skill for a file with the domain name of the
target site. Example: `playbooks/forms.example.org.md`.

If the file is available:

1. Do the freshness probe. Use `get count` on two or three known `#id` selectors from
   the playbook.
2. If all the ids are present, obey the playbook. The warm path uses approximately six
   tool calls.
3. If an id is missing, the site changed. Use the cold-start procedure. Write the
   playbook again.
4. **A partial playbook also needs the scribe.** Read the playbook for the word
   UNCONFIRMED. When the playbook marks pages or sections UNCONFIRMED and the fill
   will enter them, start the scribe (cold-start step 1) before the fill. The run
   makes new site facts in that territory, and without a scribe they are lost. A
   playbook with no UNCONFIRMED parts needs no scribe.

If the file is not available, use the cold-start procedure:

1. **Start the knowledge scribe.** The scribe is a background agent. It reads this
   session's transcript from disk and writes the playbook, the scripts, and the
   reference updates in parallel while you fill the form. The start procedure and
   the prompt template are in `references/knowledge-scribe.md`.
2. **Do not write the playbook, scripts, or references yourself during the run.**
   The scribe owns those files while it runs. You fill the form. This keeps the
   full session time on the form. If the scribe fails or is not available, finish
   the fill first and write the playbook at the end — never stop the fill to write
   documentation.
3. When the scribe's completion report arrives, relay its file list in your final
   report.

The playbook structure (the scribe follows this; you follow it in the no-scribe
fallback):

- The final form URL and the pages between the entry URL and the form
- The list of required fields, compared with the usual data payload
- A field table: id, type, maxLength, mask condition, fill method
- The gates, the effect of each gate answer, and the fill sequence
- The exact text of each select option
- The condition that enables the submit button
- Two or three stable `#id` selectors for the freshness probe

## Phase 1 — Find the Form

The URL you receive can be a landing page and not the form. Use `snapshot -i -u` to
show the links with their URLs. Follow the apply, continue, or start links until form
fields show.

Do a signal test before you fill a field. Examine the id and name prefixes of the
inputs you count. Widgets add inputs to a page. Examples: Google Translate
(`goog-gt-*`), chat widgets, cookie banners. If all the inputs belong to widgets, the
page has no form. The result of `get count input` alone is not sufficient.

## Phase 2 — Show the Gap Analysis, Then Ask the User

Read `references/gap-analysis-and-provenance.md` for the formats and the rules.

1. Make a list of the fields. Use the `FIELDS` helper in `scripts/fill-helpers.sh`.
   It gives every field with its type, id, value, and label in ONE tool call. Do not
   scan with `get count` loops. Do not use `eval`. Use selectors in this sequence of
   preference: `#id`, then `[name=…]`, then `@eN` refs. Refs become invalid after the
   DOM changes. Refs cannot identify repeated Yes/No pairs.
2. Find the required markers: `*` in labels, `required`, `aria-required`. Compare the
   required fields with the available data.
3. **Show the user the gap-analysis table first.** The table has five groups: READY,
   DERIVED, GAP-REQUIRED, GAP-DECISION, NO-FIELD. The user must see what you have,
   what you will change, and what you do not have.
4. **Each session is a new participant.** Ask for each missing required value in
   each session. Do not reuse participant answers from an earlier session. Only site
   facts persist, and they persist in the playbook.
5. **Ask for the value, not for instructions.** A gap question collects data for a
   field. Example: "What is Maria's Social Security Number?" Do not ask the user
   how to handle a missing value. The formats and the input-type rules are in
   `references/gap-analysis-and-provenance.md`.
6. Ask all the gap questions in this phase, before the fill. One `AskUserQuestion`
   call holds a maximum of four questions. If there are more gaps, use two or more
   calls in sequence. **Do not guess the answers that did not fit in one call.**
   Questions during the fill are the primary cause of a bad user experience.
7. **Write the questions in plain language.** The user completes a form. The user is
   not a developer. Do not put element ids, selectors, character limits, or browser
   terms in a question.
8. Do not invent a value for a required field. A "common baseline" or a "reasonable
   default" is an invented value. If the user gives no answer, keep the field empty
   and show it in the final report. Do not use an identifier that looks similar. A
   case number is not an SSN.
9. Treat all data as real data. Do not ask if the data is real or test data. The
   submit step always needs approval. That approval is the protection.

## Phase 3 — Fill the Fields in Gate Sequence

- Answer the master gates first. Fill from the top of the form to the bottom. A field
  behind an unanswered gate accepts commands and shows success, but the field does not
  change.
- Read the question text before you answer a gate. You cannot know the polarity
  without the text. Example: "Same as above?" with the answer "Yes" hides the related
  block. A hidden block is correct in this case. Do not fill it.
- Use `check` and `uncheck` for checkboxes. These commands are idempotent. Do not use
  `click` on a checkbox. The `click` command changes the state each time. Two clicks
  set the initial state again. Yes/No pairs are frequently independent checkboxes.
  Both can become unchecked. Read the full pair again after you write it.
- Get the list of options before you use `select`. Match the option text exactly. A
  data value of "Hispanic/Latino" does not match an option of "Hispanic". The
  `get value` command on a select returns the option index, not the text.
- Put all the write commands in one Bash call. Use `scripts/fill-helpers.sh`. Then do
  one readback of all the fields. Do not use one tool call for each field.

## Phase 4 — Make Sure That Each Write Is Correct

Read each field again after you write it. Use `get value` and `is checked`. Compare
the result with the intended value.

If a value is empty or not correct, do the four-step diagnosis in
`references/silent-failures.md`. The sequence is: disabled, hidden, masked, maxLength.
Do the steps in this sequence before you try other solutions.

Masked fields need one `key` command for each character. The `fill` and `type`
commands fail on masks that monitor keydown events. These commands show success when
they fail.

Do not use an old snapshot after the DOM changes. The ref labels become incorrect. Use
`get text` or `is visible` with a CSS selector.

## Phase 5 — Exceptions

Test a possible cause before you record it or act on it. Two incorrect diagnoses
occurred in a real session: a bot check was blamed for a disabled submit button, but
the cause was a closed affirmation section; a correctly hidden block was identified as
a defect. When the scribe runs, it records each solved exception from the transcript —
continue with the fill. Without a scribe, add the exception to the playbook at the end
of the run.

## Phase 6 — Submit Only With Approval From the User

**WARNING: Do not click submit without explicit approval from the user. A live
application has legal effect. If the values look like test data, tell the user in
one sentence in the report. The decision is the user's.**

If the submit button is disabled, find the condition that enables it on this site. Do
not assume the condition. Possible causes: a required field is empty, a consent
checkbox is not set, a disclosure section is not open, a bot challenge is not
complete. Test one possible cause at a time. Use `is enabled` after each test to find
the cause. Record the condition in the playbook.

Do not try to defeat a bot challenge (Turnstile, reCAPTCHA). Make sure that the
challenge is the cause before you identify it as the cause. If the challenge is the
cause, stop. Tell the user.

Before the submit step, show the user the provenance report. The format is in
`references/gap-analysis-and-provenance.md`. The report gives each value in the form
and its source: USER, PAYLOAD, DERIVED, or EMPTY. There is no ASSUMED source. If a
value has no traceable source, remove it from the form. After the report, list the
unused payload values, the empty required fields with reasons, and the new playbook
facts from this session. Then get explicit approval.

## Rules for New Findings

This file and `references/` contain only two types of content: agent-browser tool
behavior, and patterns confirmed on more than one site.

Write a finding from one site in the playbook of that site. Identify it as the
behavior of that site. Do not write it in this file. Move a pattern to this file only
after you see it on a different site. Write a moved pattern as a check ("check
whether…"). Do not write it as an expectation ("expect…", "usually…").

Do not put program names, site names, or real field ids from one site in this file,
in `references/`, or in `scripts/`. Those names bias the general procedure toward a
few applications. The product covers more than 100 application websites. Use neutral
examples: `forms.example.org`, `#firstName`, "Maria". Program names belong in
playbooks only.

## How to Talk to the User

Every message to the user — progress updates, reports, and questions — uses plain
language at an easy reading level (approximately grade 6). The user completes a
form. The user is not a developer.

- Do not show selectors, element ids, ref numbers, browser terms, session names, or
  tool output to the user.
- Use the person's name and the words that the form shows.
- When an agent report has a FOR THE USER section, show that section without change.
- Technical detail goes in three places only: the transcript, the playbooks, and the
  references. The scribe and the fill agents hold the technical knowledge. The main
  session translates nothing — it relays plain sections and asks plain questions.

## Tool Rules

- Use the typed commands. Do not use `eval`. The `eval` command puts unknown
  JavaScript into a third-party page. Typed commands: `is visible|enabled|checked`,
  `get value|text|attr|count`, `snapshot -i -u`, `screenshot out.png --annotate`. The
  `--annotate` option prints a legend: `[N] @eN role "name"`.
- Read `<command> --help` before you use a flag. The argument sequence of `get attr`
  is `<selector> <name>`. The full-page screenshot flag is `--full`. The tool reads an
  unknown flag as a positional argument. If a command fails, examine your syntax
  first.
- Use CSS selectors only. XPath is not supported and can give a silent no-op success.
  A selector with `[id=X]` finds only the first element with that id — see
  `references/silent-failures.md` for duplicate-id checkbox groups.
- The daemon keeps sessions with the `--session <name>` key. The `@eN` refs stay valid
  across CLI calls in one session.

## References

- `references/gap-analysis-and-provenance.md` — the two mandatory reports: the
  gap-analysis table before the fill, the provenance report before the submit step
- `references/knowledge-scribe.md` — the parallel background agent that writes the
  playbook during a cold start; start it in Phase 0
- `references/multi-application.md` — two or more applications in one run: the
  orchestrator role, scouts for cold-start discovery, one fill agent for each
  application, browser isolation, and submit gates for each application
- `references/silent-failures.md` — the four-step diagnosis with commands
- `playbooks/<domain>.md` — one file for each site; the scribe writes it on a cold
  start (you write it at the end of the run only when there is no scribe)
- `scripts/fill-helpers.sh` — helpers for batch fill, keypress fill, and readback
