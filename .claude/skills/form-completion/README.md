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

You do not set up a browser. The agent starts and controls the browser itself. By default agent-browser starts its own browser. If you want to watch the fill, tell the agent to show the browser window or to attach to your own Chrome. The agent can also attach to a remote browser provider such as Kernel or Browserbase.

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
