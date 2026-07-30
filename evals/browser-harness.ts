import { readFileSync } from "node:fs";
import { join } from "node:path";
import { tool, type Tool } from "ai";
import { z } from "zod";
import type { Command } from "agent-browser/dist/types.js";

// Mirrors the eval-facing description the agent already sees (helpers.ts).
const BROWSER_TOOL_DESCRIPTION =
  "Execute browser automation commands on a remote browser. Commands include " +
  "navigate, snapshot, click, fill, type, select, press, hover, check, uncheck, " +
  "screenshot, inputvalue, wait, evaluate, etc.";

// Mirrors the inputSchema in lib/ai/tools/browser.ts (the source of truth) — keep in sync.
const browserInputSchema = z
  .object({
    action: z.string().describe('The command action (e.g. "navigate", "click", "snapshot", "fill")'),
    selector: z.string().optional().describe("Element selector: ref (@e1), CSS (#id), or label"),
    value: z.string().optional().describe("Value for fill action"),
    text: z.string().optional().describe("Text for type action"),
    url: z.string().optional().describe("URL for navigate action"),
    key: z.string().optional().describe('Key for press action (e.g. "Enter", "Tab")'),
    label: z.string().optional().describe("Label text for getbylabel action"),
    subaction: z.string().optional().describe('Sub-action for getbylabel ("click", "fill", "check")'),
    script: z.string().optional().describe("JavaScript for evaluate action"),
    values: z.array(z.string()).optional().describe("Option values for select action — must be an array"),
    timeout: z.number().optional().describe("Timeout in ms for wait action — must be a number"),
    amount: z.number().optional().describe("Scroll amount in px — must be a number"),
    delay: z.number().optional().describe("Delay between keystrokes in ms — must be a number"),
    interactive: z.boolean().optional().describe("Show only interactive elements in snapshot — must be boolean"),
    clear: z.boolean().optional().describe("Clear field before typing — must be boolean"),
    direction: z.string().optional().describe('Scroll direction: "up" or "down"'),
    state: z.string().optional().describe('Load state for waitforloadstate (e.g. "networkidle")'),
    index: z.number().optional().describe("Tab index for tab_switch/tab_close"),
    response: z.string().optional().describe('Dialog response: "accept" or "dismiss"'),
    promptText: z.string().optional().describe("Text to enter in prompt dialog"),
  })
  .describe("Structured command object with action and action-specific parameters");

export interface BrowserSession {
  browserTool: Tool;
  /** Field values the agent submitted (review query params), with a DOM-read fallback. */
  captureSubmittedValues(): Promise<Record<string, string>>;
  close(): Promise<void>;
}

export interface CreateBrowserSessionOptions {
  /** Absolute path to a directory of fixture HTML files. */
  fixturesDir: string;
  /** Hostnames served from fixtures; all other hosts are aborted. */
  interceptHosts: string[];
}

function pathToFixtureFile(pathname: string): string {
  const clean = pathname.replace(/[.]+$/, "");
  if (clean.endsWith("/review")) return "review.html";
  return "apply.html";
}

export async function createBrowserSession(
  opts: CreateBrowserSessionOptions,
): Promise<BrowserSession> {
  const { BrowserManager } = await import("agent-browser/dist/browser.js");
  const { executeCommand } = await import("agent-browser/dist/actions.js");
  const manager = new BrowserManager();
  let n = 0;
  const exec = (params: Record<string, unknown>) =>
    executeCommand({ id: `c${n++}`, ...params } as Command, manager);

  await exec({ action: "launch", headless: true, browser: "chromium" });
  const page = manager.getPage();

  let submittedQuery: Record<string, string> = {};

  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (!opts.interceptHosts.includes(url.hostname)) {
      await route.abort();
      return;
    }
    if (url.pathname.replace(/[.]+$/, "").endsWith("/review")) {
      submittedQuery = Object.fromEntries(url.searchParams.entries());
    }
    const body = readFileSync(join(opts.fixturesDir, pathToFixtureFile(url.pathname)), "utf8");
    await route.fulfill({ status: 200, contentType: "text/html", body });
  });

  const browserTool = tool({
    description: BROWSER_TOOL_DESCRIPTION,
    inputSchema: browserInputSchema,
    execute: async (params: Record<string, unknown>) => {
      try {
        const response = await exec(params);
        if (response.success) {
          const output =
            typeof response.data === "string" ? response.data : JSON.stringify(response.data);
          return { success: true, output, error: null };
        }
        return { success: false, output: null, error: response.error };
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        return { success: false, output: null, error: message };
      }
    },
  });

  return {
    browserTool,
    captureSubmittedValues: async () => {
      if (Object.keys(submittedQuery).length > 0) return submittedQuery;
      // Fallback: agent filled the form but never submitted — read live values.
      try {
        return await page.evaluate(() => {
          const out: Record<string, string> = {};
          for (const el of Array.from(
            document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(
              "input[name], select[name], textarea[name]",
            ),
          )) {
            out[el.name] = el.value;
          }
          return out;
        });
      } catch {
        return {};
      }
    },
    close: async () => {
      await (manager as { close?: () => Promise<void> }).close?.();
    },
  };
}
