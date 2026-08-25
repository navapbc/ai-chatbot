import { describe, it, expect } from 'vitest';
import {
  getLiveViewUrl,
  setLiveViewUrl,
  clearLiveViewUrl,
} from '@/lib/ai/eve/live-view-store';

const URL_A = 'https://live.example/v/aaa';
const URL_B = 'https://live.example/v/bbb';

describe('live-view-store', () => {
  it('returns undefined for a chat with no browser yet', () => {
    expect(getLiveViewUrl('u1', 'c-unknown')).toBeUndefined();
  });

  it('stores and retrieves per (user, chat)', () => {
    setLiveViewUrl('u1', 'c1', URL_A);
    expect(getLiveViewUrl('u1', 'c1')).toBe(URL_A);
    // Isolation: the same chatId under a different user must not leak.
    expect(getLiveViewUrl('u2', 'c1')).toBeUndefined();
  });

  it('isolates chats belonging to one user', () => {
    setLiveViewUrl('u3', 'chat-a', URL_A);
    setLiveViewUrl('u3', 'chat-b', URL_B);
    expect(getLiveViewUrl('u3', 'chat-a')).toBe(URL_A);
    expect(getLiveViewUrl('u3', 'chat-b')).toBe(URL_B);
  });

  // The browser tool reports the URL on every command, so the common case is
  // writing the same value repeatedly; a genuinely new browser must win.
  it('overwrites on repeated set', () => {
    setLiveViewUrl('u4', 'c4', URL_A);
    setLiveViewUrl('u4', 'c4', URL_A);
    expect(getLiveViewUrl('u4', 'c4')).toBe(URL_A);
    setLiveViewUrl('u4', 'c4', URL_B);
    expect(getLiveViewUrl('u4', 'c4')).toBe(URL_B);
  });

  it('clears an entry', () => {
    setLiveViewUrl('u5', 'c5', URL_A);
    clearLiveViewUrl('u5', 'c5');
    expect(getLiveViewUrl('u5', 'c5')).toBeUndefined();
  });

  // /api/kernel-browser recovers chatId by trimming `-${userId}` off the
  // artifact's sessionId; this pins the arithmetic that lookup depends on.
  it('is keyed by the chatId recovered from a legacy artifact sessionId', () => {
    const userId = '95f8e324-c36c-4107-813a-90fdc6dd79b5';
    const chatId = '389fb718-feed-4da7-92a0-e440b1e79a0f';
    const artifactSessionId = `${chatId}-${userId}`;

    setLiveViewUrl(userId, chatId, URL_A);

    const recovered = artifactSessionId.slice(0, -(userId.length + 1));
    expect(recovered).toBe(chatId);
    expect(getLiveViewUrl(userId, recovered)).toBe(URL_A);
  });
});
