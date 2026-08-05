# The Knowledge Scribe — a Parallel Agent That Writes the Playbook

On a cold start, the main agent must not stop to write documentation. A second agent —
the scribe — runs in the background, reads the live session transcript from disk, and
writes the knowledge files in real time. The main agent fills the form. The scribe
writes the playbook.

## Why the Transcript, Not Messages

Claude Code writes each session to a transcript file on disk, live:
`~/.claude/projects/<project-slug>/<session-id>.jsonl`. The scribe reads this file.
The main agent does not send findings to the scribe. This costs the main agent zero
tokens and zero turns.

## How the Main Agent Starts the Scribe (Phase 0, Cold Start Only)

1. Find the transcript path. The current session file is the newest `.jsonl` in the
   project directory:

```bash
ls -t ~/.claude/projects/$(pwd | tr '/.' '--')/*.jsonl | head -1
```

If the directory is not found, list `~/.claude/projects/` and select the entry that
matches the working directory.

2. Start the scribe with the Agent tool. Run it in the background (the default). Set
   `model: "sonnet"` — the scribe reads large transcript deltas, and the smaller model
   costs approximately one third of the orchestrator model for each input token. Use
   the prompt template below. Replace `<transcript>`, `<domain>`, and `<skill-dir>`.

3. Continue with Phase 1 immediately. Do not wait for the scribe. Do not write the
   playbook, scripts, or references yourself during the run — the scribe owns those
   files while it runs. This prevents write conflicts.

4. When the scribe completes, its report arrives as a task notification. Relay the
   list of written files in your final report (report item 3).

## Scribe Prompt Template

```
You are the knowledge scribe for a live form-completion session. The main agent
fills a web form for <domain>. Your task is to read the transcript of the main
agent and to write the knowledge files. You do not fill the form. You do not talk
to the user.

Transcript: <transcript>
Skill directory: <skill-dir>   (contains playbooks/, references/, scripts/)

## Loop

1. Read the new transcript bytes since your last pass:
   OFFSET=0 at start; then: tail -c +$((OFFSET+1)) "<transcript>"; update OFFSET
   with: wc -c < "<transcript>".
2. Distill findings from the tool calls and results (see "What to Extract").
3. Write or update the files.
4. Wait approximately 60 seconds. Then repeat.
5. Stop when one of these is true: the transcript shows the provenance report or a
   submit decision; or the file has no new bytes for 5 minutes. Do one final pass,
   then return your report.

## What to Extract, and Where It Goes

- SITE FACTS -> <skill-dir>/playbooks/<domain>.md
  The URL chain to the real form. Field ids, types, maxLength values, mask methods.
  Gates, their polarity, and the fill sequence. Exact select option text. Duplicate-id
  groups with their value/label maps. The submit enablement condition. Two or three
  stable ids for the freshness probe. Follow the playbook structure in SKILL.md
  Phase 0. Write in ASD-STE100 Simplified Technical English.
- REUSABLE SHELL PATTERNS -> <skill-dir>/scripts/
  If the main agent improvises the same shell pattern two or more times, write it as
  a helper function with STE comments. Check the syntax with `bash -n`. Do not add a
  helper for a pattern seen one time on one site.
- TOOL SEMANTICS -> append to <skill-dir>/references/silent-failures.md General Rules
  Only behaviors of agent-browser itself (a flag that no-ops, a selector class that
  fails). These are site-independent. Site behaviors NEVER go in references — they go
  in the playbook. This is the anti-overfitting rule in SKILL.md.

## Hard Rules

- NEVER run agent-browser commands. The browser session belongs to the main agent.
  A command from you can change the form state during the fill. You read the
  transcript and you write files. Do nothing else.
- NEVER write participant data into any file. The transcript contains names, phone
  numbers, addresses, and possibly an SSN. Playbook examples use placeholders
  ("MMDDYYYY", "the 2-letter state code", "input[value='N']"), never the
  participant's values.
- Record a finding when the transcript CONFIRMS it (a readback, an is-check, a
  count). Do not record a finding when the main agent only tries something. A failed
  try is a finding only when the transcript shows the correct alternative with it.
- Update, do not duplicate: if the playbook file exists, merge into it.

## Your Report (returned when you stop)

- Files written or updated, one line each with what changed
- Findings you saw but did NOT record, with the reason (unconfirmed, participant
  data, single-site pattern)
```

## Multi-Application Runs

In a multi-application run (`multi-application.md`), the scribe is the only writer
of ALL knowledge files, and this includes the playbooks. Fill agents and scouts
write no files. Each scout report and each fill-agent report arrives in the
orchestrator transcript with a SITE FACTS section: the URL chain, the field table,
masks, gates, option texts, and the submit condition. The scribe writes the first
playbook for a cold-start domain from the SCOUT report — the URL chain, the account
requirement, the field tables, and the unreached sections (keep the UNCONFIRMED
marks). When the fill-agent report for that domain arrives, merge it into the same
playbook: confirmed methods replace UNCONFIRMED entries. The scribe also merges the
findings about tool behavior into the references. The scribe stop condition changes
in this mode: stop after the LAST fill-agent report is in the transcript, not after
the first.

## Failure Behavior

- If the scribe stops or does not start, the fill continues with no damage. The main
  agent completes the fill. The playbook is written on the next cold start. Do not
  stop the fill because of a scribe problem.
- If the scribe and the main agent find the same fact, the scribe writes it. The
  main agent does not write files during the run, so a conflict is not possible.

## Cost

The scribe is one background agent that reads a local file each minute. It reads
only the new bytes in each pass, not the full file. The cost is approximately the
tokens of one more short session. The main run spends no turns on documentation,
and the playbook is complete when the run ends.
