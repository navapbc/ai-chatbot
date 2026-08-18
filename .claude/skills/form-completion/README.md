# Form Completion Skill

This skill gives Claude Code a tested procedure to complete web forms with
[agent-browser](https://www.npmjs.com/package/agent-browser). It covers
application forms, benefits forms, and multi-step apply flows. Each rule in the
procedure comes from a failure that occurred in a real session — silent fill
failures, masked date fields, gated sections, and checkbox polarity traps.

## What You Get

- `SKILL.md` — the six-phase fill procedure (find playbook → find form → gap
  analysis → fill → verify → submit with approval)
- `references/` — deep guides for silent-failure diagnosis, gap-analysis and
  provenance reports, the knowledge-scribe background agent, and
  multi-application orchestration
- `playbooks/` — one file per site with confirmed selectors and gates; the
  skill writes a new playbook for each new site it completes
- `scripts/fill-helpers.sh` — bash helpers for batch fill, masked-field
  keypress fill, and value readback in one tool call

## Install the Skill

The fastest path is the [skills CLI](https://github.com/vercel-labs/skills). It
works with Claude Code and 70+ other agents:

```bash
npx skills add navapbc/ai-chatbot --skill form-completion
```

Or install manually. Copy this directory into your project or user skills
directory:

```bash
# Project scope (this project only)
git clone --depth 1 https://github.com/navapbc/ai-chatbot /tmp/ai-chatbot
cp -r /tmp/ai-chatbot/.claude/skills/form-completion .claude/skills/

# User scope (all your projects)
cp -r /tmp/ai-chatbot/.claude/skills/form-completion ~/.claude/skills/
```

Claude Code loads the skill automatically when a task matches its
description. You can also invoke it directly with `/form-completion`.

## Dependency: agent-browser

The skill does not work without the `agent-browser` CLI. It is a native Rust
binary distributed through npm. Install it in one of two ways:

```bash
# Global — the binary is on your PATH
npm install -g agent-browser

# Project-local — the binary is at ./node_modules/.bin/agent-browser
npm install agent-browser
```

The skill and its helpers look for the binary at
`./node_modules/.bin/agent-browser` by default. If you installed it globally
or somewhere else, point the helpers at it:

```bash
export AGENT_BROWSER_BIN=agent-browser
```

Version `0.33` or later is required. Earlier versions do not have the daemon
`--session` flag that keeps element refs valid across CLI calls.

## Dependency: a Browser to Drive

agent-browser attaches to a Chromium browser over CDP. Pick one:

**Local Chrome (recommended for watching the fill).** Start Chrome with remote
debugging, then attach:

```bash
# macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 --user-data-dir="$HOME/.chrome-debug" &

agent-browser --session form-fill --cdp http://localhost:9222 open https://example.org
```

**Bundled browser.** With no `--cdp` flag, agent-browser launches its own
browser. Add `--headed` to watch it.

**Remote browser provider** (Kernel.sh, Browserbase, etc.). Pass the
provider's CDP WebSocket URL with `--cdp <ws_url>`.

Use one `--session <name>` value for the whole run. The daemon holds the CDP
connection between calls, so element refs (`@e1`) stay valid across commands.

## Quick Start

1. Install the skill and agent-browser (above).
2. Start Claude Code in your project.
3. Ask: *"Fill out the application at https://forms.example.org for Maria
   Garcia"* — include the applicant data in the message or point at a file.

The skill then:

1. Checks `playbooks/` for the target domain and probes that it is fresh
2. Navigates from the landing page to the actual form
3. Shows you a **gap-analysis table** — what data it has, what it derived, and
   what it needs from you — and asks for missing values before it fills
4. Fills in gate order, then reads every field back to catch silent failures
5. Shows a **provenance report** (every value and its source) and **stops for
   your explicit approval before it submits**

The skill never clicks submit on its own, never invents values for required
fields, and never attempts to defeat a bot challenge.

## Notes for Maintainers

- General patterns confirmed on two or more sites go in `SKILL.md` or
  `references/`. Single-site facts go in that site's playbook only. See "Rules
  for New Findings" in `SKILL.md`.
- `scripts/fill-helpers.sh` is bash. On Windows, run it through WSL or Git
  Bash.
