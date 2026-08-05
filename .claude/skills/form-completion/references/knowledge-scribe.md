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

5. After the submit decision (Phase 6), send the scribe ONE message that contains
   the word FINALIZE (SendMessage, with the scribe's agent id). This is the
   scribe's stop signal. Without it, the scribe guesses the end of the run from
   stall timeouts, and it guesses badly: a fill can pause for many minutes while
   the user answers questions.

## Scribe Prompt Template

```
You are the knowledge scribe for a live form-completion session. The main agent
fills a web form for <domain>. Your task is to read the transcript of the main
agent and to write the knowledge files. You do not fill the form. You do not talk
to the user.

Transcript: <transcript>
Skill directory: <skill-dir>   (contains playbooks/, references/, scripts/)

## Loop — Wake on Markers, Not on Minutes

Each of your turns reads your full conversation again. A 60-second poll loop in one
run made 606 turns and read 125 million cached tokens. Your budget for the full run
is approximately 40 turns. Obey these rules:

1. NEVER call a tool to pass the time. No sleep, no "echo waiting", no size checks
   while you wait. When there is no new marker, END YOUR TURN. The Monitor
   notification starts your next turn.
2. Load the Monitor tool (ToolSearch "select:Monitor"). Start ONE Monitor with the
   script below, non-persistent. The script is silent until the new bytes hold a
   marker, then it prints one line and exits. One notification for each wake.
3. On a wake: read the new bytes ONCE (tail -c +$((OFFSET+1)); update OFFSET with
   wc -c). Extract. Write the files. Start the Monitor again with the new offset.
   End the turn.
4. The Monitor's byte count and your own read can differ. A partial line write
   causes this. Do NOT investigate the difference. Parse complete JSON lines only
   and continue.
5. Finalize when a message with the word FINALIZE arrives from the main agent, or
   when the Monitor prints STALL (the backstop). Then do ONE final pass: the leak
   check (Hard Rules), then your report.

Monitor script (replace <transcript> and <offset>):

   F="<transcript>"; OFF=<offset>; CYCLES=0
   while true; do
     sleep 30
     CUR=$(wc -c < "$F")
     if [ "$CUR" -gt "$OFF" ]; then
       if tail -c +$((OFF+1)) "$F" | grep -qE "SITE FACTS|task-notification|FINALIZE"; then
         echo "MARKER"; exit 0
       fi
       OFF=$CUR
     fi
     CYCLES=$((CYCLES+1))
     [ "$CYCLES" -ge 40 ] && { echo "STALL"; exit 0; }
   done

Growth without a marker is intake dialogue — participant answers, not site facts.
The script skips it without a wake. Your read on the next wake covers those bytes.

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
  participant's values. A value that equals a participant value from the transcript
  IS participant data, also when it looks like an example: the 9 digits that the
  transcript shows in an SSN fill are the participant's entry — write "NNN-NN-NNNN"
  in the file. Before your final report, run the leak check: grep every knowledge
  file for the participant's names, numbers, dates, and address parts that you saw
  in the transcript. Remove each hit. Your report includes the leak-check result.
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
findings about tool behavior into the references. The stop signal does not change:
the orchestrator sends FINALIZE after the last submit decision. Do not stop on a
fill-agent report — a BLOCKED application continues after the user answers, and one
application can report five or more times.

## Failure Behavior

- If the scribe stops or does not start, the fill continues with no damage. The main
  agent completes the fill. The playbook is written on the next cold start. Do not
  stop the fill because of a scribe problem.
- If the scribe and the main agent find the same fact, the scribe writes it. The
  main agent does not write files during the run, so a conflict is not possible.

## Cost

The content the scribe ingests is small. The turns are the cost: each turn reads
the scribe's full conversation again at cache-read prices. A poll-loop scribe in
one run made 606 turns and read 125 million cached tokens — more than the fill
agents that did the work. The marker-wake loop above makes approximately one turn
for each report, near 40 turns for a three-application run. The main run still
spends no turns on documentation, and the playbook is complete when the run ends.
