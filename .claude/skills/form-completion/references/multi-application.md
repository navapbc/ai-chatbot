# Multi-Application Runs — One Orchestrator, One Fill Agent for Each Application

Use this procedure when one session must complete two or more applications (example:
a county program plus a state program). The unit of parallel work is the APPLICATION,
not the form section. Applications have independent browsers, so they can run in
parallel. Sections of one form share one page, so they cannot.

## Roles

| Role | Count | Job |
|---|---|---|
| Orchestrator | 1 (the main session) | All user conversation. Consolidated intake. Dispatch. The consolidated final report. Submit approvals. |
| Scout | 1 for each cold-start application | Find the real form. Return the field inventory. No fills. No account creation. No user conversation. No file writes. |
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
- **The scribe is the only writer of knowledge files.** Fill agents and scouts write
  no files. They return their findings in the SITE FACTS section of their reports. The
  reports arrive in the orchestrator transcript, and the scribe writes the playbook
  from them. One writer keeps the language rules, the no-bias rule, and the
  no-participant-data rule in one place.
- **The orchestrator does not survey a site.** After the tab setup and the freshness
  probes, the orchestrator sends no agent-browser commands. A survey by the
  orchestrator stops the intake: the user waits, and the gap analysis for the ready
  applications does not start. This occurred on 2026-08-05: the orchestrator followed
  a large portal's apply links itself, and the gap analysis for two applications with
  fresh playbooks did not start. A scout does the survey.

## Procedure

### 1. Consolidated Intake (Orchestrator)

1. For each application: find the playbook, do the freshness probe (Phase 0 of
   SKILL.md).
2. For an application with NO playbook, start a scout agent in the background, now
   (see "The Scout" below). Do not survey the site yourself — the hard rule above.
   You cannot make a correct gap analysis for a form that no agent read, so the
   scout reads it while you talk to the user.
3. Build the gap analysis for the applications WITH playbooks, from their playbook
   field tables. Show it and ask the questions NOW. Do not wait for the scouts.
4. Ask the gap questions grouped by application, in plain language (the rules in
   `gap-analysis-and-provenance.md` apply). Put the application name in the header
   of each question (example: "IHSS: SSN", "WIC: clinic"). The user answers
   questions for two or more forms in one list, so the header is the only signal of
   which form a question belongs to. For a scouted application, tell the user what
   comes next (example: "The food-assistance application needs more answers — I
   will ask them when the survey of that site is complete"). Do not ask a question
   that the scout did not confirm.
5. When a scout report arrives, build the gap analysis for that application from
   the report, and ask a second round of questions. A large application can have
   20 or more gaps — obey "Large Forms" in `gap-analysis-and-provenance.md`: show
   the full table first, then ask in groups that follow the sections of the form.
6. Report a submit blocker in the gap analysis, with the gaps. Examples: a reCAPTCHA
   on the form, or an account wall from a scout report. The fill can finish, but the
   submit needs the user. The user must know this before the fill starts, not after.
7. Do not start a fill agent before its application has all its required answers.
   Start the applications that are ready. The warm applications usually start while
   the scouts run.

#### The Scout (Cold-Start Discovery)

A scout is a background agent that reads ONE application site and returns the field
inventory. Start it with the Agent tool, `model: "sonnet"` (it must find the real
form behind menus and interstitial pages, and that needs judgment). The scout uses
the session name and the tab that the fill agent will use later, so the browser is
already on the form when the fill agent starts. Scout prompt template:

```
Survey ONE application site for a form fill that a different agent will do later.
Do not fill any field. Do not enter any data. Do not create an account. Do not
talk to a user. Do not write files.

- Application: <url>
- Browser: from <client dir> run `./node_modules/.bin/agent-browser --session
  <name> <cmd>`. The session is ALREADY connected and the tab is ALREADY open.
  Your FIRST command is `tab <label>`; then `get url` to confirm the selection.
  Do NOT run `connect`. Do NOT run `tab new`.
- Helpers: `source <skill-dir>/scripts/fill-helpers.sh; SESSION=<name>` gives
  SURVEY, FIELDS, IFRAMES, and OPTIONS.

Procedure:
1. Find the real form. Follow the apply or start links through the interstitial
   pages. Record the URL chain.
2. If the site asks for an account or a login before the form, stop there. Report
   it. Do not make an account.
3. On each form page that you can reach: run SURVEY, then FIELDS. Record the ids,
   the types, the labels, the required marks, the maxlength values, and the exact
   select option texts.
4. A multi-page form hides the later pages until data goes in. Do not enter data
   to reach them. Report the pages that you reached. When the page shows its own
   section list (a progress bar, a menu, a table of contents), record the section
   names and mark each unreached section UNCONFIRMED.
5. Leave the browser on the first form page (or on the account wall). The fill
   agent continues from there.

Return SITE FACTS only (this report is for the orchestrator and the scribe, not
for a user):
- STATUS: FORM-REACHED, ACCOUNT-WALL, or NOT-FOUND
- The URL chain
- The account or login requirement, if one exists
- The field table for each page that you reached
- The section names for the pages that you did not reach, each marked UNCONFIRMED
- Bot checks that you saw (do not try to pass one)
- Approximate tool call count
```

The scout report arrives as a task notification. It also lands in the orchestrator
transcript, so the scribe writes the first playbook for the domain from it
(`knowledge-scribe.md`). The fill-agent prompt for that application then contains
the scout's field table as its field map.

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

- Watchable one-window mode — use this when the user wants ONE window with one tab
  for each application. Two sessions can share one debug Chrome when each session
  owns its own tab. This is confirmed by test on 2026-08-05: the two sessions sent
  keystrokes and fills in alternation, and each value landed in the correct tab. No
  value landed in the wrong tab. Do the steps in this sequence:

```bash
# SETUP PHASE — the orchestrator does ALL tab creation, before any fill starts.
# The `tab new` command moves the tab pointer of EVERY session in the window.
# A `tab new` during a fill sends the next keystrokes into the wrong application.
agent-browser --session <app1> connect 9222
agent-browser --session <app1> tab new --label <app1> <url1>
agent-browser --session <app2> connect 9222
agent-browser --session <app2> tab new --label <app2> <url2>

# FILL PHASE — the FIRST command of each fill agent selects its own tab:
agent-browser --session <app1> tab <app1>
agent-browser --session <app1> get url        # make sure that the selection is correct
```

  Obey these rules in this mode. No agent runs `tab new` after the fills start.
  Each fill agent selects its tab by label as its first command, and reads the URL
  to confirm the selection. The one-writer rule applies to the TAB, not to the
  window. The applications share the browser cookies. This is acceptable for
  applications on different domains. It is not acceptable for two applications on
  one portal that permits one session for each login. Close each tab with
  `tab close <label>`. Do not use `close`.

Do not connect two writer sessions to one port without the tab procedure above.
Two sessions that connect to one port attach to the same active tab (confirmed by
test), and two writers on one tab put text in each other's fields.

### 3. Dispatch (Orchestrator)

**Select the model for each fill agent.** The Agent tool takes a `model` parameter.
Without it, the agent inherits the orchestrator model, which costs approximately ten
times more for each input token. The fill agents read the large DOM outputs, so the
model choice controls most of the run cost.

| Agent | Model parameter | Reason |
|---|---|---|
| Fill agent with a playbook (warm path) | `haiku` | The playbook gives the exact selectors and methods. The fill is checklist work. |
| Fill agent with no playbook (cold start) | `sonnet` | Discovery needs judgment: label mapping, gate polarity. |
| Scout | `sonnet` | It must find the real form behind menus and interstitial pages. |
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
- Field map (cold start only): <the field table from the scout report, plus the
  fields that the agent must NOT fill and why. The browser is already ON the form
  page — the scout left it there.>
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

- A scout dies or finds no form: report it to the user with the scout's last known
  state. The other applications continue. Do not survey the site yourself.
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
