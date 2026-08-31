// Judge config shared by every online scorer.

// Runs inside Braintrust, so this resolves against the org's configured AI
// providers — not lib/ai/providers.ts. The org currently has only a direct
// Anthropic key, so judges do NOT go through Vertex. Once a Google Vertex AI
// provider exists, repoint this one constant (the qualified
// `publishers/anthropic/models/...` form may be required).
export const JUDGE_MODEL = 'claude-sonnet-5';

// Only a clean pass (1.0) counts; 0.5 means "minor issues".
export const PASS_THRESHOLD = 0.75;
