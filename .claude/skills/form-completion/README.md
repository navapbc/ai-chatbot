[![skills.sh](https://skills.sh/b/navapbc/ai-chatbot)](https://skills.sh/navapbc/ai-chatbot)

An agent skill that completes web forms with [agent-browser](https://github.com/vercel-labs/agent-browser). It covers application forms, benefits forms, and multi-step apply flows. Each rule in the procedure comes from a failure that occurred in a real session.

## Skills

- [**form-completion**](SKILL.md): A six-phase procedure to complete a web form. The skill finds the form, shows a gap analysis, asks the user for missing data, fills the fields, verifies each write, and waits for approval from the user before it submits. It includes references for silent-failure diagnosis, site playbooks with confirmed selectors, and bash helpers for batch fill and readback.

## Install

### CLI

Works in Claude Code, Codex, Opencode and other agents.

```bash
npx skills add navapbc/ai-chatbot
```

### Claude Code plugin

Installs the skill and updates in place. Run these inside Claude Code:

```text
/plugin marketplace add navapbc/ai-chatbot
/plugin install form-completion@ai-chatbot
```

## Requirements

### agent-browser

The skill does not work without the agent-browser CLI, version 0.33 or later. It is a native Rust binary on npm.

```bash
npm install -g agent-browser
```

A project-local installation also works. The helper scripts look for the binary at `./node_modules/.bin/agent-browser`. If the binary is at a different location, set the path:

```bash
export AGENT_BROWSER_BIN=agent-browser
```

### A browser

agent-browser attaches to a Chromium browser through CDP. Use one of these options:

Local Chrome. This option lets you watch the fill. Start Chrome with remote debugging, then attach:

```bash
# macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 --user-data-dir="$HOME/.chrome-debug" &

agent-browser --session form-fill --cdp http://localhost:9222 open https://example.org
```

Bundled browser. With no `--cdp` flag, agent-browser starts its own browser. Add `--headed` to watch it.

Remote browser provider (Kernel, Browserbase, and others). Pass the provider CDP WebSocket URL with `--cdp <ws_url>`.

Use one `--session <name>` value for the full run. The daemon holds the CDP connection between calls, so element refs such as `@e1` stay valid across commands.

## Use

Ask your agent to fill a form and give it the applicant data:

```text
Fill out the application at https://forms.example.org for Maria Garcia.
```

The skill then does these steps:

1. Checks `playbooks/` for the target domain and makes sure the playbook is fresh
2. Navigates from the landing page to the form
3. Shows a gap-analysis table with the data it has, the data it derived, and the data it needs, then asks for the missing values before it fills
4. Fills the fields in gate order, then reads each field again to catch silent failures
5. Shows a provenance report with the source of each value and stops for explicit approval before it submits

The skill never clicks submit on its own, never invents values for required fields, and never tries to defeat a bot challenge.

## Notes for Maintainers

- General patterns confirmed on two or more sites go in `SKILL.md` or `references/`. Facts from one site go in the playbook of that site only. See "Rules for New Findings" in `SKILL.md`.
- `scripts/fill-helpers.sh` is bash. On Windows, run it through WSL or Git Bash.
