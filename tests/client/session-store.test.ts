import { expect, test } from 'vitest';
import {
  buildSessionStatus,
  cacheKey,
  isProfileUsable,
  profileNameFor,
  type SessionLike,
} from '@/lib/kernel/session-store';

test('cacheKey namespaces by user then session', () => {
  expect(cacheKey('user-1', 'chat-1-user-1')).toBe('user-1:chat-1-user-1');
});

test('profileNameFor sanitizes to Kernel-allowed characters', () => {
  // sessionId is `${chatId}-${userId}`; both are usually uuid-ish already.
  expect(profileNameFor('chat_1.2-user-1')).toBe('sess-chat_1.2-user-1');
  // Disallowed chars (spaces, slashes, colons) collapse to hyphens.
  expect(profileNameFor('a b/c:d')).toBe('sess-a-b-c-d');
});

test('profileNameFor caps length at 255 chars', () => {
  const long = 'x'.repeat(400);
  expect(profileNameFor(long).length).toBe(255);
});

const liveSession: SessionLike = {
  liveViewUrl: 'https://live.example/view',
  startedAt: 1000,
  lastActivityAt: 2000,
  standby: false,
};

test('buildSessionStatus reports a missing session', () => {
  const status = buildSessionStatus(undefined, 5000);
  expect(status).toEqual({
    exists: false,
    standby: false,
    liveViewUrl: null,
    startedAt: 0,
    lastActivityAt: 0,
    now: 5000,
  });
});

test('buildSessionStatus exposes the live view while connected', () => {
  const status = buildSessionStatus(liveSession, 5000);
  expect(status.exists).toBe(true);
  expect(status.standby).toBe(false);
  expect(status.liveViewUrl).toBe('https://live.example/view');
  expect(status.startedAt).toBe(1000);
  expect(status.lastActivityAt).toBe(2000);
  expect(status.now).toBe(5000);
});

test('buildSessionStatus never leaks a live view URL while in standby', () => {
  const status = buildSessionStatus({ ...liveSession, standby: true }, 5000);
  expect(status.exists).toBe(true);
  expect(status.standby).toBe(true);
  // The browser is paused — handing back its URL would point at a dead view.
  expect(status.liveViewUrl).toBeNull();
});

test('isProfileUsable: created (no error) is usable', () => {
  expect(isProfileUsable(undefined)).toBe(true);
});

test('isProfileUsable: 409 conflict (already exists) is usable', () => {
  expect(isProfileUsable(409)).toBe(true);
});

test('isProfileUsable: other errors are not usable (fall back to no profile)', () => {
  // 400 is the "profile not found" class of failure that broke creation;
  // 500/403 etc. should likewise degrade gracefully rather than block.
  expect(isProfileUsable(400)).toBe(false);
  expect(isProfileUsable(403)).toBe(false);
  expect(isProfileUsable(500)).toBe(false);
});
