// Judge config shared by every online scorer.

// Runs inside Braintrust, so this resolves against the org's configured AI
// providers — not lib/ai/providers.ts. The org currently has only a direct
// Anthropic key, so judges do NOT go through Vertex. Once a Google Vertex AI
// provider exists, repoint this one constant (the qualified
// `publishers/anthropic/models/...` form may be required).
export const JUDGE_MODEL = 'claude-sonnet-5';

// UNVERIFIED — confirm before activating any Jev rule.
//
// Braintrust supports Jev as a judge model, but only documents picking it by
// name in the Scorers UI; it does not publish the identifier the REST API
// wants, and api.braintrust.dev was returning 502 when this was written, so
// it could not be read back from /v1. This value is the TypeSafe SDK's own
// default model id, which is the most likely spelling but is a guess.
//
// To confirm: create a throwaway LLM-judge scorer in the Braintrust UI, pick
// Jev in the model picker, save it, then GET /v1/function and read the saved
// `prompt_data.options.model`. Repoint this one constant to match.
//
// Prerequisite, separate from this constant: an org Owner must turn on
// Settings > AI providers > Allow built-in models, then Enable Jev. Built-in
// Jev is free and needs no TypeSafe account; an org-scoped TypeSafe provider
// using our own TYPESAFE_API_KEY is the alternative.
export const JEV_JUDGE_MODEL = 'jev-latest';

// Only a clean pass (1.0) counts; 0.5 means "minor issues".
export const PASS_THRESHOLD = 0.75;
