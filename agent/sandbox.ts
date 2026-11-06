import { defineSandbox } from 'eve/sandbox';

// Representative sandbox. Omitting `backend` uses defaultBackend() — Vercel
// Sandbox on Vercel, else Docker/microsandbox/just-bash locally.
//
// Two relevant uses for THIS agent:
//  1. The browser-automation skill's sibling reference files
//     ($HOME/.agents/skills/browser-automation/*) are read through the sandbox
//     at runtime via ctx.getSkill(...).file(...).
//  2. Browser automation itself runs today via Kernel.sh as an app-runtime tool
//     (Eve tools run in the app runtime, NOT the sandbox). A future Eve-native
//     port could instead run headless Chromium inside this sandbox — see
//     docs/eve-spike-findings.md "Browser session sketch" (sub-project 3).
export default defineSandbox({
  async onSession({ use }) {
    // Per-session setup would go here (network policy, credentials). Kept
    // minimal for the demonstrative conversion.
    await use();
  },
});
