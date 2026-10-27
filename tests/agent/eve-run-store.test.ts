import { randomUUID } from 'node:crypto';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

// queries.ts builds its Postgres client from DATABASE_URL at module load, and
// ESM imports are hoisted, so point it at the scratch database first.
vi.hoisted(() => {
  if (process.env.TEST_DATABASE_URL) {
    process.env.DATABASE_URL = process.env.TEST_DATABASE_URL;
  }
});
vi.mock('server-only', () => ({}));

import postgres from 'postgres';
import { createDbRunStore } from '@/lib/ai/eve/db-run-store';
import type { PendingInputRequest } from '@/lib/ai/eve/input-requests';
import {
  createMemoryRunStore,
  EveRunExistsError,
  type EveRunStore,
} from '@/lib/ai/eve/run-store';
import type { RunProgress } from '@/lib/relay/progress';

const TEST_DATABASE_URL = process.env.TEST_DATABASE_URL;

const pending: PendingInputRequest[] = [
  {
    requestId: 'r1',
    prompt: 'Which county?',
    display: 'select',
    options: [{ id: 'o1', label: 'Alameda' }],
    allowFreeform: false,
  },
];

const progress: RunProgress = {
  sections: [{ name: 'Household', status: 'filling', filled: 2, total: 5 }],
  current: 'Household',
};

function runSuite(
  makeStore: () => EveRunStore,
  users: () => { owner: string; other: string },
) {
  const init = (token = 't1') => ({
    eveSessionId: 's1',
    continuationToken: token,
  });

  it('create then get round trips', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    const created = await store.create(owner, chatId, init());
    const expected = {
      eveSessionId: 's1',
      continuationToken: 't1',
      streamIndex: 0,
      relayOrigin: null,
      pendingRequests: [],
      progress: null,
      endedAt: null,
      turnIdle: false,
    };
    expect(created).toEqual(expected);
    expect(await store.get(owner, chatId)).toEqual(expected);
  });

  it("get for another user's chat returns undefined", async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());
    expect(await store.get(other, chatId)).toBeUndefined();
  });

  it('create twice rejects with EveRunExistsError', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init('t1'));
    await expect(store.create(owner, chatId, init('t2'))).rejects.toBeInstanceOf(
      EveRunExistsError,
    );
    expect((await store.get(owner, chatId))?.continuationToken).toBe('t1');
  });

  it('swapToken succeeds with the right expected token only', async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init('t1'));

    expect(await store.swapToken(owner, chatId, 'wrong', 't2')).toBe(false);
    expect((await store.get(owner, chatId))?.continuationToken).toBe('t1');

    // Right token, wrong user: still scoped by userId.
    expect(await store.swapToken(other, chatId, 't1', 't2')).toBe(false);
    expect((await store.get(owner, chatId))?.continuationToken).toBe('t1');

    expect(await store.swapToken(owner, chatId, 't1', 't2')).toBe(true);
    expect((await store.get(owner, chatId))?.continuationToken).toBe('t2');
  });

  it('advance moves only the cursor, only forward, and never writes a token', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init('t1'));

    await store.advance(owner, chatId, { streamIndex: 5 });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 't1',
      streamIndex: 5,
    });

    await store.advance(owner, chatId, { streamIndex: 5 });
    await store.advance(owner, chatId, { streamIndex: 3 });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 't1',
      streamIndex: 5,
    });

    await store.advance(owner, chatId, { streamIndex: 9 });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 't1',
      streamIndex: 9,
    });
  });

  it("advance heals the turn's own claim with the turn's token", async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init('t1'));
    await store.swapToken(owner, chatId, 't1', 'claim:a');

    await store.advance(owner, chatId, {
      streamIndex: 4,
      heal: { claim: 'claim:a', token: 't2' },
    });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 't2',
      streamIndex: 4,
    });
  });

  it('advance heals a claim even on a replayed (lower) boundary, cursor kept', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init('t1'));
    await store.advance(owner, chatId, { streamIndex: 8 });
    await store.swapToken(owner, chatId, 't1', 'claim:a');

    await store.advance(owner, chatId, {
      streamIndex: 6,
      heal: { claim: 'claim:a', token: 't2' },
    });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 't2',
      streamIndex: 8,
    });
  });

  it('advance never replaces a token that is not the given claim', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    // The turn's final swap already stored its token: a heal for that turn
    // must not touch it, whatever index the boundary has.
    await store.create(owner, chatId, init('t3'));
    await store.advance(owner, chatId, {
      streamIndex: 7,
      heal: { claim: 'claim:a', token: 't2' },
    });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 't3',
      streamIndex: 7,
    });
    // Another turn's claim is not this turn's either.
    await store.swapToken(owner, chatId, 't3', 'claim:b');
    await store.advance(owner, chatId, {
      streamIndex: 9,
      heal: { claim: 'claim:a', token: 't2' },
    });
    expect(await store.get(owner, chatId)).toMatchObject({
      continuationToken: 'claim:b',
      streamIndex: 9,
    });
  });

  it('setPending and setProgress round trip and survive get', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());

    await store.setPending(owner, chatId, pending);
    await store.setProgress(owner, chatId, progress);
    const got = await store.get(owner, chatId);
    expect(got?.pendingRequests).toEqual(pending);
    expect(got?.progress).toEqual(progress);

    await store.setPending(owner, chatId, []);
    expect((await store.get(owner, chatId))?.pendingRequests).toEqual([]);
  });

  it('getByChatId returns the owner and relayOrigin', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, {
      ...init(),
      relayOrigin: 'https://benefits.example.gov',
    });
    expect(await store.getByChatId(chatId)).toMatchObject({
      userId: owner,
      relayOrigin: 'https://benefits.example.gov',
      eveSessionId: 's1',
    });
    expect(await store.getByChatId(randomUUID())).toBeUndefined();
  });

  it('getByChatId ignores web rows, so one never hides a relay run', async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    await store.create(other, chatId, init());
    expect(await store.getByChatId(chatId)).toBeUndefined();
    await store.create(owner, chatId, {
      ...init(),
      relayOrigin: 'https://benefits.example.gov',
    });
    expect(await store.getByChatId(chatId)).toMatchObject({ userId: owner });
  });

  it('a second relay run on one chatId throws, for any user', async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    const relay = { ...init(), relayOrigin: 'https://benefits.example.gov' };
    await store.create(owner, chatId, relay);
    await expect(store.create(other, chatId, relay)).rejects.toBeInstanceOf(
      EveRunExistsError,
    );
    await expect(store.create(owner, chatId, relay)).rejects.toBeInstanceOf(
      EveRunExistsError,
    );
  });

  it('countRecentRelayRuns counts only that user\'s relay rows since the cutoff', async () => {
    const store = makeStore();
    const { owner, other } = users();
    const relay = { ...init(), relayOrigin: 'https://benefits.example.gov' };
    const before = new Date(Date.now() - 1000);
    const baseline = await store.countRecentRelayRuns(owner, before);
    await store.create(owner, randomUUID(), relay);
    await store.create(owner, randomUUID(), relay);
    await store.create(owner, randomUUID(), init());
    await store.create(other, randomUUID(), relay);
    expect(await store.countRecentRelayRuns(owner, before)).toBe(baseline + 2);
    expect(
      await store.countRecentRelayRuns(owner, new Date(Date.now() + 60_000)),
    ).toBe(0);
  });

  it('setPending, setProgress, advance and remove are no-ops for a missing run', async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    await store.setPending(owner, chatId, pending);
    await store.setProgress(owner, chatId, progress);
    await store.advance(owner, chatId, {
      streamIndex: 9,
      heal: { claim: 't1', token: 't9' },
    });
    await store.remove(owner, chatId);
    expect(await store.get(owner, chatId)).toBeUndefined();
    expect(await store.getByChatId(chatId)).toBeUndefined();

    // Writes scoped to another user leave the owner's run untouched.
    await store.create(owner, chatId, init('t1'));
    await store.setPending(other, chatId, pending);
    await store.setProgress(other, chatId, progress);
    await store.advance(other, chatId, {
      streamIndex: 9,
      heal: { claim: 't1', token: 't9' },
    });
    await store.remove(other, chatId);
    expect(await store.get(owner, chatId)).toEqual({
      eveSessionId: 's1',
      continuationToken: 't1',
      streamIndex: 0,
      relayOrigin: null,
      pendingRequests: [],
      progress: null,
      endedAt: null,
      turnIdle: false,
    });
    expect(await store.get(other, chatId)).toBeUndefined();
  });

  it('remove deletes', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());
    await store.remove(owner, chatId);
    expect(await store.get(owner, chatId)).toBeUndefined();
  });

  it('end returns true once, then false, and the stored time does not move', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());

    expect(await store.end(owner, chatId)).toBe(true);
    const first = (await store.get(owner, chatId))?.endedAt;
    expect(first).toBeInstanceOf(Date);

    await new Promise((resolve) => setTimeout(resolve, 15));
    expect(await store.end(owner, chatId)).toBe(false);
    const second = (await store.get(owner, chatId))?.endedAt;
    expect(second?.getTime()).toBe(first?.getTime());
  });

  it('get and getByChatId expose endedAt', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, {
      ...init(),
      relayOrigin: 'https://app.example.test',
    });
    expect((await store.get(owner, chatId))?.endedAt).toBeNull();
    expect((await store.getByChatId(chatId))?.endedAt).toBeNull();

    await store.end(owner, chatId);
    const viaGet = (await store.get(owner, chatId))?.endedAt;
    const viaChat = (await store.getByChatId(chatId))?.endedAt;
    expect(viaGet).toBeInstanceOf(Date);
    expect(viaChat?.getTime()).toBe(viaGet?.getTime());
  });

  it('end on an unknown run returns false and creates nothing', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    expect(await store.end(owner, chatId)).toBe(false);
    expect(await store.get(owner, chatId)).toBeUndefined();
  });

  it("end for another user's chat id returns false and leaves the row unended", async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());
    expect(await store.end(other, chatId)).toBe(false);
    expect((await store.get(owner, chatId))?.endedAt).toBeNull();
    expect(await store.get(other, chatId)).toBeUndefined();
  });

  it('swapToken with live: true refuses an ended row and keeps its token', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init('t1'));
    expect(
      await store.swapToken(owner, chatId, 't1', 't2', { live: true }),
    ).toBe(true);
    expect(await store.end(owner, chatId)).toBe(true);
    expect(
      await store.swapToken(owner, chatId, 't2', 't3', { live: true }),
    ).toBe(false);
    expect((await store.get(owner, chatId))?.continuationToken).toBe('t2');
    // Without the option an ended row still swaps (rollback, final swap).
    expect(await store.swapToken(owner, chatId, 't2', 't3')).toBe(true);
    expect((await store.get(owner, chatId))?.continuationToken).toBe('t3');
  });

  it('a web row can also be ended', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());
    expect(await store.end(owner, chatId)).toBe(true);
    expect((await store.get(owner, chatId))?.endedAt).toBeInstanceOf(Date);
  });

  it('create starts turnIdle false; setTurnIdle sets it true then false', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    const created = await store.create(owner, chatId, {
      ...init(),
      relayOrigin: 'https://forms.example.gov',
    });
    expect(created.turnIdle).toBe(false);

    await store.setTurnIdle(owner, chatId, true);
    expect((await store.get(owner, chatId))?.turnIdle).toBe(true);
    expect((await store.getByChatId(chatId))?.turnIdle).toBe(true);

    await store.setTurnIdle(owner, chatId, false);
    expect((await store.get(owner, chatId))?.turnIdle).toBe(false);
  });

  it('setTurnIdle on an unknown run is a no-op and creates nothing', async () => {
    const store = makeStore();
    const { owner } = users();
    const chatId = randomUUID();
    await store.setTurnIdle(owner, chatId, true);
    expect(await store.get(owner, chatId)).toBeUndefined();
  });

  it("setTurnIdle for another user's chat id leaves the owner's row untouched", async () => {
    const store = makeStore();
    const { owner, other } = users();
    const chatId = randomUUID();
    await store.create(owner, chatId, init());
    await store.setTurnIdle(other, chatId, true);
    expect((await store.get(owner, chatId))?.turnIdle).toBe(false);
    expect(await store.get(other, chatId)).toBeUndefined();
  });
}

describe('memory run store', () => {
  const ids = { owner: randomUUID(), other: randomUUID() };
  runSuite(createMemoryRunStore, () => ids);
});

if (!TEST_DATABASE_URL) {
  console.info(
    '[eve-run-store] TEST_DATABASE_URL is not set; skipping the Drizzle run store suite.',
  );
}

describe.skipIf(!TEST_DATABASE_URL)('drizzle run store', () => {
  const ids = { owner: randomUUID(), other: randomUUID() };
  let sql: ReturnType<typeof postgres>;

  beforeAll(async () => {
    sql = postgres(TEST_DATABASE_URL as string, { max: 1 });
    await sql`insert into "User" (id, email) values
      (${ids.owner}, ${`${ids.owner}@test.local`}),
      (${ids.other}, ${`${ids.other}@test.local`})`;
  });

  afterAll(async () => {
    await sql`delete from "EveRun" where "userId" in (${ids.owner}, ${ids.other})`;
    await sql`delete from "User" where id in (${ids.owner}, ${ids.other})`;
    await sql.end();
  });

  runSuite(createDbRunStore, () => ids);

  it('create for a nonexistent user rejects with the FK error, not EveRunExistsError', async () => {
    const store = createDbRunStore();
    const error = await store
      .create(randomUUID(), randomUUID(), {
        eveSessionId: 's1',
        continuationToken: 't1',
      })
      .catch((e: unknown) => e);
    expect(error).not.toBeInstanceOf(EveRunExistsError);
    expect(error).toMatchObject({
      code: '23503',
      constraint_name: 'EveRun_userId_User_id_fk',
    });
  });

  it('concurrent swapToken with the same expected token: exactly one wins', async () => {
    const store = createDbRunStore();
    const chatId = randomUUID();
    await store.create(ids.owner, chatId, {
      eveSessionId: 's1',
      continuationToken: 'base',
    });
    const nexts = Array.from({ length: 10 }, (_, i) => `next-${i}`);
    const results = await Promise.all(
      nexts.map((n) => store.swapToken(ids.owner, chatId, 'base', n)),
    );
    const winners = nexts.filter((_, i) => results[i]);
    expect(winners).toHaveLength(1);
    expect((await store.get(ids.owner, chatId))?.continuationToken).toBe(
      winners[0],
    );
  });

  it('racing advance calls leave the max index and the token untouched', async () => {
    const store = createDbRunStore();
    const chatId = randomUUID();
    await store.create(ids.owner, chatId, {
      eveSessionId: 's1',
      continuationToken: 't0',
    });
    const indices = Array.from({ length: 20 }, (_, i) => i + 1).sort(
      () => Math.random() - 0.5,
    );
    await Promise.all(
      indices.map((i) => store.advance(ids.owner, chatId, { streamIndex: i })),
    );
    expect(await store.get(ids.owner, chatId)).toMatchObject({
      streamIndex: 20,
      continuationToken: 't0',
    });
  });

  it('racing heals of one claim: the token is written once, cursor is the max', async () => {
    const store = createDbRunStore();
    const chatId = randomUUID();
    await store.create(ids.owner, chatId, {
      eveSessionId: 's1',
      continuationToken: 'claim:x',
    });
    await Promise.all(
      Array.from({ length: 10 }, (_, i) =>
        store.advance(ids.owner, chatId, {
          streamIndex: i + 1,
          heal: { claim: 'claim:x', token: 'healed' },
        }),
      ),
    );
    expect(await store.get(ids.owner, chatId)).toMatchObject({
      streamIndex: 10,
      continuationToken: 'healed',
    });
  });
});
