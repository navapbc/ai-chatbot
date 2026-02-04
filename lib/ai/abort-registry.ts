import { Redis } from '@upstash/redis';

// Initialize Redis for distributed abort signal storage
// This is critical for Cloud Run where multiple instances may handle requests
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

const REDIS_KEY_PREFIX = 'chat-abort:';
const ABORT_TTL_SECONDS = 5 * 60; // 5 minutes

// In-memory registry for local AbortControllers (per-instance)
// When an abort signal is received via Redis pub/sub or polling, we use this to actually abort
const globalForAbort = globalThis as typeof globalThis & {
  abortControllers?: Map<string, AbortController>;
};

if (!globalForAbort.abortControllers) {
  globalForAbort.abortControllers = new Map<string, AbortController>();
}

const localAbortControllers = globalForAbort.abortControllers;

function getRedisKey(chatId: string): string {
  return `${REDIS_KEY_PREFIX}${chatId}`;
}

/**
 * Register a new AbortController for a chat.
 * If there's an existing one, it will be aborted first.
 * Returns the new AbortController's signal.
 */
export async function registerAbortController(chatId: string): Promise<AbortController> {
  // First, signal any previous request to abort (distributed)
  await signalAbort(chatId);

  // Abort any local controller
  const existingController = localAbortControllers.get(chatId);
  if (existingController) {
    existingController.abort();
  }

  // Create new controller
  const controller = new AbortController();
  localAbortControllers.set(chatId, controller);

  // Mark this chat as active in Redis
  await redis.set(getRedisKey(chatId), { active: true, timestamp: Date.now() }, { ex: ABORT_TTL_SECONDS });

  return controller;
}

/**
 * Signal that a chat should be aborted.
 * This sets a flag in Redis that can be polled by other instances.
 */
export async function signalAbort(chatId: string): Promise<void> {
  // Mark as aborted in Redis
  await redis.set(getRedisKey(chatId), { aborted: true, timestamp: Date.now() }, { ex: ABORT_TTL_SECONDS });

  // Abort local controller if exists
  const controller = localAbortControllers.get(chatId);
  if (controller) {
    controller.abort();
    localAbortControllers.delete(chatId);
  }
}

/**
 * Check if a chat has been signaled to abort.
 * Used for polling in long-running operations.
 */
export async function isAborted(chatId: string): Promise<boolean> {
  const data = await redis.get<{ aborted?: boolean }>(getRedisKey(chatId));
  return data?.aborted === true;
}

/**
 * Clean up the abort controller for a chat.
 * Called when the request completes normally.
 */
export async function cleanupAbortController(chatId: string): Promise<void> {
  localAbortControllers.delete(chatId);
  await redis.del(getRedisKey(chatId));
}

/**
 * Get the local AbortController for a chat (if running on this instance).
 */
export function getLocalAbortController(chatId: string): AbortController | undefined {
  return localAbortControllers.get(chatId);
}
