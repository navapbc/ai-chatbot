import { readFileSync } from "node:fs";
import { join } from "node:path";
import { chromium, type Browser, type Page } from "playwright";
import { tool, type Tool } from "ai";
import { browserInputSchema } from "@/lib/ai/tools/browser";
import { ACTION_TIMEOUT_MS, executeCliCommand } from "./browser-commands";

// Mirrors the eval-facing description the agent already sees (helpers.ts).
const BROWSER_TOOL_DESCRIPTION =
  "Execute an agent-browser command on a browser. Pass the command as an argv " +
  'array, e.g. ["open", "<url>"], ["snapshot"], ["click", "@e1"], ' +
  '["fill", "@e1", "text"], ["get", "value", "@e1"], ["eval", "<js>"].';

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
  // agent-browser is a native binary that drives a remote Kernel browser; evals
  // run offline against fixtures, so Playwright provides the local page and
  // `executeCliCommand` interprets the agent's argv against it.
  const browser: Browser = await chromium.launch({ headless: true });
  const page: Page = await browser.newPage();
  page.setDefaultTimeout(ACTION_TIMEOUT_MS);
  // Ref map lives for the session, mirroring the CLI daemon's.
  const refs = new Map<string, string>();

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
    execute: async ({ command }: { command: string[] }) =>
      executeCliCommand(page, refs, command),
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
      await browser.close();
    },
  };
}
