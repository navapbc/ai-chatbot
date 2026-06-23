import assert from "node:assert";
import { fileURLToPath } from "node:url";
import { createBrowserSession } from "./browser-harness";

async function main() {
  const fixturesDir = fileURLToPath(new URL("./fixtures/calfresh", import.meta.url));
  const session = await createBrowserSession({
    fixturesDir,
    interceptHosts: ["calfresh.example.gov"],
  });
  // The execute signature is (input, options); options is ignored by the harness.
  const run = (p: Record<string, unknown>) =>
    (session.browserTool as any).execute(p, {}) as Promise<{
      success: boolean;
      output: string | null;
      error: string | null;
    }>;

  try {
    const nav = await run({ action: "navigate", url: "https://calfresh.example.gov/apply" });
    assert.ok(nav.success, "navigate should succeed");

    const snap = await run({ action: "snapshot", interactive: true });
    assert.ok(snap.output, "snapshot should return data");
    assert.match(snap.output, /Age/, "snapshot lists the Age field");
    assert.match(snap.output, /Mailing Address/, "snapshot lists Mailing Address");

    const refs = JSON.parse(snap.output).refs as Record<string, { role: string; name: string }>;
    const ageRef = Object.entries(refs).find(([, v]) => v.name === "Age")?.[0];
    assert.ok(ageRef, "Age field has a ref");

    const fill = await run({ action: "fill", selector: `@${ageRef}`, value: "37" });
    assert.ok(fill.success, "fill age should succeed");

    const read = await run({ action: "inputvalue", selector: `@${ageRef}` });
    assert.match(read.output ?? "", /37/, "age value persists in the live page");

    const nextRef = Object.entries(refs).find(([, v]) => /next/i.test(v.name))?.[0];
    assert.ok(nextRef, "Next button has a ref");
    const click = await run({ action: "click", selector: `@${nextRef}` });
    assert.ok(click.success, "clicking Next should succeed");

    const submitted = await session.captureSubmittedValues();
    assert.strictEqual(submitted.age, "37", "captured submitted age reflects the fill");

    // network isolation: a non-intercepted host must not load
    const ext = await run({ action: "navigate", url: "https://example.com/" });
    assert.ok(!ext.success, "navigating to a non-intercepted host should fail");

    console.log("HARNESS CHECK PASSED");
  } finally {
    await session.close();
  }
}

main().catch((err) => {
  console.error("HARNESS CHECK FAILED:", err);
  process.exit(1);
});
