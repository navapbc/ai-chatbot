# Multi-Application Runs — One Orchestrator, One Fill Agent for Each Application

Use this procedure when one session must complete two or more applications (example:
a county program plus a state program). The unit of parallel work is the APPLICATION,
not the form section. Applications have independent browsers, so they can run in
parallel. Sections of one form share one page, so they cannot.

## Roles

| Role | Count | Job |
|---|---|---|
| Orchestrator | 1 (the main session) | All user conversation. Consolidated intake. Dispatch. The consolidated final report. Submit approvals. |
| Fill agent | 1 for each application | Fill and verify ONE application in its own browser. No user conversation. No file writes. |
| Scribe | 1 (background) | Reads the orchestrator transcript. Writes ALL knowledge files: `playbooks/`, `scripts/`, `references/`. See `knowledge-scribe.md`. |

## Hard Rules

- **One writer for each page. This rule has no exceptions.** When two agents write
  to one tab, each agent moves the keyboard focus. Keystrokes from one agent land in
  the field that the other agent selected, and every command shows success
  (confirmed by test on 2026-08-05). Each fill agent gets its own agent-browser
  session name and its own browser instance.
- **Only the orchestrator talks to the user.** A fill agent that needs an answer
  stops and returns a BLOCKED report. It never waits for a user.
- **The scribe is the only writer of knowledge files.** Fill agents write no files.
  A fill agent returns its findings in the SITE FACTS section of its report. The
  report arrives in the orchestrator transcript, and the scribe writes the playbook
  from it. One writer keeps the language rules, the no-bias rule, and the
  no-participant-data rule in one place.

## Procedure

### 1. Consolidated Intake (Orchestrator)

1. For each application: find the playbook, do the freshness probe (Phase 0 of
   SKILL.md).
2. For an application with NO playbook, do the field inventory yourself, now, before
   the gap analysis. You cannot make a correct gap analysis for a form that you did
   not read. Open the page in the session that the fill agent will use, then:
   - `get count` for `input`, `select`, `textarea`, `iframe`, `form` — this gives the
     size of the form and shows if an iframe can hold it.
   - `FIELDS "body"` (see `scripts/fill-helpers.sh`) — the full field map in one call.
   - Read the `src` of each iframe. Many iframes are reCAPTCHA and ad trackers. Do
     not assume that the form is inside an iframe. On the WIC form, all 6 iframes
     were noise and the form was inline.
   - Probe `required` and `maxlength` on the fields that you plan to fill.
   Put this map in the fill-agent prompt. The fill agent then starts with a plan, not
   with discovery. The scribe writes the playbook from the reports at the end.
3. Build ONE gap analysis that covers all the applications. Group the gaps by
   application.
4. Ask the user all the gap questions now, grouped by application, in plain language
   (the rules in `gap-analysis-and-provenance.md` apply). Tell the user which
   application each question group belongs to. Put the application name in the header
   of each question (example: "IHSS: SSN", "WIC: clinic"). The user answers questions
   for two forms in one list, so the header is the only signal of which form a
   question belongs to.
5. Report a submit blocker in the gap analysis, with the gaps. Example: a reCAPTCHA on
   the form. The fill can finish, but the submit needs the user. The user must know
   this before the fill starts, not after.
6. Do not start a fill agent before its application has all its required answers.
   Start the applications that are ready. Ask the user about the other applications
   while the started agents run.

### 2. Browser Isolation

Give each fill agent its own session name. Use the domain as the name
(`--session forms-example-org`, `--session apply-example-gov`).

- Default mode: each fill agent opens its own browser with `open <url>`. Each
  agent-browser session starts its own browser. The browsers share no state.
- Watchable local mode — use this when the user wants to see the fills live. Start
  one debug Chrome for each application, each with its own port and its own profile:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=<9222+N> --user-data-dir=/tmp/ab-debug-profile-<N> \
  --no-first-run --no-default-browser-check &
# then: agent-browser --session <name> connect <9222+N>
```

Do not connect two writer sessions to one port. Two sessions on one browser attach
to the same active tab (confirmed by test). Do not put two applications in two tabs
of one browser: the applications share the browser cookies, and a portal that
permits one session for each login can cancel tab 1 when tab 2 sends a request.

### 3. Dispatch (Orchestrator)

**Select the model for each fill agent.** The Agent tool takes a `model` parameter.
Without it, the agent inherits the orchestrator model, which costs approximately ten
times more for each input token. The fill agents read the large DOM outputs, so the
model choice controls most of the run cost.

| Agent | Model parameter | Reason |
|---|---|---|
| Fill agent with a playbook (warm path) | `haiku` | The playbook gives the exact selectors and methods. The fill is checklist work. |
| Fill agent with no playbook (cold start) | `sonnet` | Discovery needs judgment: label mapping, gate polarity. |
| Scribe | `sonnet` | It writes the knowledge files that later runs depend on. |

A small-model fill agent stays safe because of the BLOCKED rule: when the playbook
does not cover a situation, the agent returns BLOCKED and does not improvise. The
orchestrator supplies the judgment. If a small-model agent returns wrong fills or
invented values, record it and use `sonnet` for that path.

Start all the ready fill agents in ONE message (parallel tool calls). Each fill
agent prompt contains:

```
Fill ONE application. Do not talk to a user. Follow the form-completion skill
(.claude/skills/form-completion/SKILL.md), phases 1, 3, 4, and 6, with these inputs:

- Application: <url — the DIRECT form url, not a landing page>
- Playbook: <path — "read it first, the freshness probe passed, obey it" — or
  "none, cold start">
- Browser: from <client dir> run `./node_modules/.bin/agent-browser --session <name>
  <cmd>`. The session is ALREADY connected to port <N> and the page is ALREADY open.
  Do NOT run `connect` again. Do NOT use port <M> — another agent owns it.
- Helpers: `source <skill-dir>/scripts/fill-helpers.sh; SESSION=<name>` gives
  S/K/C/U/V/FIELDS. Use K for each masked field. Use FIELDS for a same-id group.
- Field map (cold start only): <the inventory from step 1.2, plus the fields that
  the agent must NOT fill and why>
- Answered gap table: <the values and decisions from intake, for this application.
  Give the exact selector and the exact value for each field. Say which checkbox
  values stay clear.>
- Payload values with no form field: <the list> — report these as unused. Do not
  force them into a field that looks similar.
- Skill directory: <path>

Rules:
- Phase 2 is complete. Do not ask questions. If you find a NEW gap during the fill,
  leave the field empty, mark the application BLOCKED on that field, and continue
  with the other fields.
- Do not guess a value. A field with no value in the table above stays empty and goes
  in the report as EMPTY. This includes a Yes/No pair: leave both boxes clear.
- Verify every write (Phase 4). Do not submit (Phase 6 approval happens in the main
  session).
- Batch the writes: many write commands in ONE Bash call, then ONE readback pass. Do
  not use one tool call for each field.
- Write NO files. The scribe writes the playbook from your report.

Return this report, in two sections:

FOR THE USER (plain language, easy reading level, two to four sentences):
- What is complete, what is empty, and what needs the user. Use the person's name
  and the words on the form. No selectors, no ids, no browser terms, no counts of
  tool calls.

SITE FACTS (technical, for the scribe):
- STATUS: COMPLETE or BLOCKED
- The provenance table for this application (format: gap-analysis-and-provenance.md)
- The fields that you left empty, with the reason for each
- The submit-gate state: the button state, the bot-check token state, and if a human
  must do a step before the submit
- BLOCKED fields with the question the user must answer, in plain language
- Site facts found (corrections to an existing playbook are also site facts): the
  URL chain, the field table, masks, gates, option texts, duplicate-id groups, the
  submit condition, freshness-probe ids
- Approximate tool call count
```

Do not put the paths of the knowledge files in the prompt. Fill agents write no
files, so they do not need the paths.

Result of one run with this prompt shape: two applications, in parallel, with 10 and 13
tool calls, and no silent failure on the first readback pass. The warm application (a
128-input form with a playbook) did all 25 writes in ONE Bash call.

### 4. Collect, Resolve, Continue (Orchestrator)

- Fill agent reports arrive as task notifications. Show the user the FOR THE USER
  section of each report, without change. Do not show the SITE FACTS section — the
  scribe consumes it.
- BLOCKED report: ask the user the returned questions (plain language, grouped),
  then continue THE SAME agent with SendMessage and the answers. Do not start a new
  fill agent — the running agent holds the browser session and the context.
- One failed application does not stop the others. Show the partial results in the
  report.

### 5. Consolidated Report and Submit Gates (Orchestrator)

- Show one provenance report for each application (the format in
  `gap-analysis-and-provenance.md`), plus the total tool calls across all agents.
- **Each application has its own submit gate.** Ask for approval one application at
  a time. Approval for one application is not approval for the others. On approval,
  tell the fill agent to submit with SendMessage; it verifies the submit result and
  reports back.

## Failure Behavior

- A fill agent dies: report that application as failed with its last known state.
  The other applications continue. Offer the user a restart of the failed one.
- The scribe dies: no effect on the fills (see `knowledge-scribe.md`).
- A browser stops during a fill: the fill agent reports BLOCKED with the list of
  verified fields. A new fill agent uses the playbook and does the fill again with
  few steps.

## What This Does Not Cover

Parallel agents on the SECTIONS of one form. Sections share one page and one
keyboard focus. That fan-out gains no time and can corrupt the fill. See the
single-application procedure in SKILL.md.
