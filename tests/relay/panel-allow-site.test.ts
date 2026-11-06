import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ALLOW_TAB_POLL_MS,
  type AllowSiteDeps,
  type AllowSiteView,
  createAllowSiteController,
  showAllowSite,
} from '../../extension/panel/lib/allow-site';
import { ALLOW_SITE_COPY } from '../../extension/panel/lib/copy';
import { UNKNOWN_PAGE } from '../../extension/panel/lib/tab-url';
import { WorkerError } from '../../extension/panel/lib/worker-api';

// The allow-this-site card's controller against fakes, on vitest's fake
// clock. It gates a security control: the agent may follow the tab to
// another origin only after the caseworker presses Allow, so these tests pin
// that nothing but allow() ever reaches the worker's allowOrigin, and that a
// press on a hold that moved or ended does nothing.

const ROOT = path.resolve(__dirname, '../..');
const NOW = 1_800_000_000_000;
const A = 'https://login.example.gov';
const B = 'https://idp.example.gov';
const TAB = 7;

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (v: T) => void;
  reject: (e: unknown) => void;
};
const deferred = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const setup = (
  opts: { origin?: string; holdEndsAt?: number; url?: string } = {},
) => {
  const origin = opts.origin ?? A;
  const plan = {
    url: opts.url ?? `${origin}/sso/start?case=Zelda#frag`,
    tab: undefined as (() => Promise<{ url?: string } | undefined>) | undefined,
    allow: async (): Promise<void> => {},
  };
  const allowOrigin = vi.fn(async (o: string) => {
    void o;
    return plan.allow();
  });
  const getTab = vi.fn(async (id: number) => {
    expect(id).toBe(TAB);
    if (plan.tab) return plan.tab();
    return { url: plan.url };
  });
  const onAllowed = vi.fn();
  const deps: AllowSiteDeps = {
    allowOrigin,
    getTab,
    onAllowed,
    now: () => Date.now(),
    setInterval: (fn, ms) => setInterval(fn, ms),
    clearInterval: (id) => clearInterval(id),
  };
  const views: AllowSiteView[] = [];
  const controller = createAllowSiteController(
    { origin, tabId: TAB, holdEndsAt: opts.holdEndsAt },
    deps,
    (v) => views.push(v),
  );
  controller.update({ pendingOrigin: origin, stopping: false });
  const last = () => views[views.length - 1] ?? controller.view();
  return { controller, allowOrigin, getTab, onAllowed, plan, views, last };
};

const tick = (ms: number) => vi.advanceTimersByTimeAsync(ms);

beforeEach(() => {
  vi.useFakeTimers({ now: NOW });
});
afterEach(() => {
  vi.useRealTimers();
});

describe('showAllowSite', () => {
  it('shows for a pending origin on a live run', () => {
    expect(showAllowSite({ status: 'running', pendingOrigin: A })).toBe(A);
  });

  it('hides without a pending origin, or once the run ended', () => {
    expect(showAllowSite({ status: 'running' })).toBeNull();
    expect(showAllowSite({ status: 'running', pendingOrigin: '' })).toBeNull();
    expect(showAllowSite({ status: 'ended', pendingOrigin: A })).toBeNull();
    expect(showAllowSite(null)).toBeNull();
  });
});

describe('allow-site controller: what it shows', () => {
  it('names the pending origin and reads the tab at start', async () => {
    const t = setup();
    expect(t.last().origin).toBe(A);
    expect(t.last().tabText).toBeNull();
    t.controller.start();
    await tick(0);
    expect(t.getTab).toHaveBeenCalledTimes(1);
    // Origin plus path: never the query or the fragment.
    expect(t.last().tabText).toBe(`${A}/sso/start`);
    expect(JSON.stringify(t.views)).not.toMatch(/Zelda|frag/);
    expect(t.last()).toMatchObject({ phase: 'waiting', canAllow: true });
    t.controller.dispose();
  });

  it('reads the tab again every 2 s and follows it', async () => {
    const t = setup();
    expect(ALLOW_TAB_POLL_MS).toBe(2000);
    t.controller.start();
    await tick(0);
    t.plan.url = `${A}/mfa?x=1`;
    await tick(2000);
    expect(t.getTab).toHaveBeenCalledTimes(2);
    expect(t.last().tabText).toBe(`${A}/mfa`);
    t.controller.dispose();
  });

  it('a failed tab read shows an unknown page', async () => {
    const t = setup();
    t.plan.tab = async () => {
      throw new Error('No tab with id: 7');
    };
    t.controller.start();
    await tick(0);
    expect(t.last().tabText).toBe(UNKNOWN_PAGE);
    t.plan.tab = async () => undefined;
    await tick(2000);
    expect(t.last().tabText).toBe(UNKNOWN_PAGE);
    t.plan.tab = async () => ({});
    await tick(2000);
    expect(t.last().tabText).toBe(UNKNOWN_PAGE);
    t.plan.tab = async () => ({ url: 'chrome://newtab/' });
    await tick(2000);
    expect(t.last().tabText).toBe(UNKNOWN_PAGE);
    t.controller.dispose();
  });

  it('skips a tab read while one is in flight; a late answer after dispose is dropped', async () => {
    const t = setup();
    const slow = deferred<{ url?: string }>();
    t.plan.tab = () => slow.promise;
    t.controller.start();
    await tick(10_000);
    expect(t.getTab).toHaveBeenCalledTimes(1);
    const before = t.views.length;
    t.controller.dispose();
    slow.resolve({ url: `${A}/late` });
    await tick(0);
    expect(t.views.length).toBe(before);
  });

  it('a known countdown counts down to 0 and never disables Allow', async () => {
    const t = setup({ holdEndsAt: NOW + 60_000 });
    t.controller.start();
    await tick(0);
    expect(t.last().secondsLeft).toBe(60);
    await tick(1000);
    expect(t.last().secondsLeft).toBe(59);
    await tick(58_500);
    expect(t.last().secondsLeft).toBe(1);
    await tick(500);
    expect(t.last().secondsLeft).toBe(0);
    // Informational only: the worker decides whether the hold is still on.
    expect(t.last().canAllow).toBe(true);
    await tick(10_000);
    expect(t.last().secondsLeft).toBe(0);
    expect(t.last().canAllow).toBe(true);
    t.controller.dispose();
  });

  it('an unknown countdown shows none', async () => {
    const t = setup();
    t.controller.start();
    await tick(30_000);
    expect(t.last().secondsLeft).toBeNull();
    expect(t.last().canAllow).toBe(true);
    t.controller.dispose();
  });

  it('countdown copy', () => {
    expect(ALLOW_SITE_COPY.secondsLeft(60)).toBe('About 60 seconds left');
    expect(ALLOW_SITE_COPY.secondsLeft(1)).toBe('About 1 second left');
    expect(ALLOW_SITE_COPY.holdMayHaveEnded).toBe('The hold may have ended');
    expect(ALLOW_SITE_COPY.tooMany).toBe(
      'You have already allowed the most sites for this run.',
    );
    expect(ALLOW_SITE_COPY.unreachable).toBe(
      'The extension did not answer. Try again.',
    );
    expect(ALLOW_SITE_COPY.moved(A)).toBe(
      `The page moved to ${A}. The agent is paused until you decide.`,
    );
    expect(ALLOW_SITE_COPY.tabNow(`${A}/x`)).toBe(`The tab is now on: ${A}/x`);
    expect(ALLOW_SITE_COPY.allow).toBe('Allow this site for this run');
    expect(ALLOW_SITE_COPY.stop).toBe('Stop the run');
    expect(ALLOW_SITE_COPY.holdEnded).toBe(
      'The hold ended. The agent has stopped waiting.',
    );
  });

  it('Allow is disabled while the run is stopping', async () => {
    const t = setup();
    t.controller.update({ pendingOrigin: A, stopping: true });
    expect(t.last().canAllow).toBe(false);
    await t.controller.allow();
    expect(t.allowOrigin).not.toHaveBeenCalled();
    t.controller.dispose();
  });
});

describe('allow-site controller: Allow', () => {
  it('calls allowOrigin once with the card origin, then reports allowed', async () => {
    const t = setup();
    t.controller.start();
    await t.controller.allow();
    expect(t.allowOrigin).toHaveBeenCalledTimes(1);
    expect(t.allowOrigin).toHaveBeenCalledWith(A);
    expect(t.onAllowed).toHaveBeenCalledTimes(1);
    expect(t.last()).toMatchObject({
      phase: 'allowed',
      canAllow: false,
      notice: null,
    });
    t.controller.dispose();
  });

  it('a second press while in flight, or after success, sends nothing', async () => {
    const t = setup();
    const pending = deferred<void>();
    t.plan.allow = () => pending.promise;
    const first = t.controller.allow();
    // Same tick: the latch is synchronous.
    const second = t.controller.allow();
    expect(t.last()).toMatchObject({ phase: 'allowing', canAllow: false });
    pending.resolve();
    await first;
    await second;
    await t.controller.allow();
    expect(t.allowOrigin).toHaveBeenCalledTimes(1);
    expect(t.onAllowed).toHaveBeenCalledTimes(1);
    t.controller.dispose();
  });

  it('a press after the pending origin changed sends nothing', async () => {
    const t = setup();
    t.controller.update({ pendingOrigin: B, stopping: false });
    expect(t.last().canAllow).toBe(false);
    await t.controller.allow();
    expect(t.allowOrigin).not.toHaveBeenCalled();
    // The parent remounts the card for B; this one only stays inert.
    expect(t.last()).toMatchObject({ phase: 'waiting', notice: null });
    t.controller.dispose();
  });

  it('a press after the pending origin cleared sends nothing', async () => {
    const t = setup();
    t.controller.update({ pendingOrigin: undefined, stopping: false });
    await t.controller.allow();
    expect(t.allowOrigin).not.toHaveBeenCalled();
    expect(t.last().notice).toBeNull();
    t.controller.dispose();
  });

  it('an expired countdown does not block Allow: the worker decides', async () => {
    const t = setup({ holdEndsAt: NOW + 5000 });
    t.controller.start();
    await tick(20_000);
    expect(t.last()).toMatchObject({ secondsLeft: 0, canAllow: true });
    await t.controller.allow();
    expect(t.allowOrigin).toHaveBeenCalledTimes(1);
    expect(t.last().phase).toBe('allowed');
    t.controller.dispose();
  });

  it('an expired countdown then a refusal: the hold ended', async () => {
    const t = setup({ holdEndsAt: NOW + 5000 });
    t.plan.allow = async () => {
      throw new WorkerError('origin_not_pending');
    };
    t.controller.start();
    await tick(6000);
    await t.controller.allow();
    expect(t.last()).toMatchObject({
      phase: 'refused',
      notice: 'holdEnded',
      canAllow: false,
    });
    t.controller.dispose();
  });

  it.each([
    ['origin not pending', new WorkerError('origin_not_pending'), 'holdEnded'],
    ['no active run', new WorkerError('no_active_run'), 'holdEnded'],
    ['wrong run', new WorkerError('wrong_run'), 'holdEnded'],
    ['worker error', new WorkerError('worker_error'), 'holdEnded'],
    ['a plain error', new Error('boom'), 'holdEnded'],
    ['too many origins', new WorkerError('too_many_origins'), 'tooMany'],
  ])(
    'a refusal (%s) shows its fixed message and disables Allow',
    async (_n, e, notice) => {
      const t = setup();
      t.plan.allow = async () => {
        throw e;
      };
      await t.controller.allow();
      expect(t.allowOrigin).toHaveBeenCalledTimes(1);
      expect(t.onAllowed).not.toHaveBeenCalled();
      expect(t.last()).toMatchObject({
        phase: 'refused',
        notice,
        canAllow: false,
      });
      await t.controller.allow();
      expect(t.allowOrigin).toHaveBeenCalledTimes(1);
      t.controller.dispose();
    },
  );

  it('an unreachable worker says so and leaves Allow for a retry press', async () => {
    const t = setup();
    t.plan.allow = async () => {
      throw new WorkerError('worker_unreachable');
    };
    t.controller.start();
    await t.controller.allow();
    expect(t.last()).toMatchObject({
      phase: 'waiting',
      notice: 'unreachable',
      canAllow: true,
    });
    // Never retried by itself: polls and timers send nothing more.
    for (let i = 0; i < 20; i++) {
      t.controller.update({ pendingOrigin: A, stopping: false });
      await tick(1000);
    }
    expect(t.allowOrigin).toHaveBeenCalledTimes(1);
    // A second press is a new decision: it is sent, and clears the notice.
    t.plan.allow = async () => {};
    await t.controller.allow();
    expect(t.allowOrigin).toHaveBeenCalledTimes(2);
    expect(t.last()).toMatchObject({ phase: 'allowed', notice: null });
    t.controller.dispose();
  });

  it('success after the card was removed still reports allowed, without a view', async () => {
    const t = setup();
    const pending = deferred<void>();
    t.plan.allow = () => pending.promise;
    const press = t.controller.allow();
    // The worker poll cleared the hold before the reply arrived.
    t.controller.dispose();
    const before = t.views.length;
    pending.resolve();
    await press;
    expect(t.onAllowed).toHaveBeenCalledTimes(1);
    expect(t.views.length).toBe(before);
  });

  it('allow() after dispose sends nothing', async () => {
    const t = setup();
    t.controller.dispose();
    await t.controller.allow();
    expect(t.allowOrigin).not.toHaveBeenCalled();
  });
});

describe('allow-site controller: never auto-approved', () => {
  it('100 polls, tab changes, origin changes and timers never call allowOrigin', async () => {
    const t = setup({ holdEndsAt: NOW + 60_000 });
    t.controller.start();
    const urls = [
      `${A}/a?x=1`,
      `${B}/b#y`,
      'chrome://settings/',
      `${A}/allow`,
      'https://evil.example/Allow this site for this run',
    ];
    for (let i = 0; i < 100; i++) {
      t.plan.url = urls[i % urls.length] ?? '';
      const pendingOrigin = [A, B, undefined, A][i % 4];
      t.controller.update({ pendingOrigin, stopping: i % 7 === 0 });
      await tick(700);
    }
    // Back to the card's origin, with time to spare: still only a press.
    t.controller.update({ pendingOrigin: A, stopping: false });
    await tick(120_000);
    expect(t.allowOrigin).not.toHaveBeenCalled();
    expect(t.onAllowed).not.toHaveBeenCalled();
    expect(t.getTab.mock.calls.length).toBeGreaterThan(30);
    t.controller.dispose();
  });

  it('only allow() reaches allowOrigin, and only the button calls allow()', () => {
    const panel = path.join(ROOT, 'extension/panel');
    const files: string[] = [];
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const full = path.join(dir, name);
        if (statSync(full).isDirectory()) walk(full);
        else if (/\.(ts|tsx)$/.test(name)) files.push(full);
      }
    };
    walk(panel);
    const callers = files
      .filter((f) => /allowOrigin\s*\(/.test(readFileSync(f, 'utf8')))
      .map((f) => path.relative(ROOT, f))
      .sort();
    // worker-api.ts declares it; allow-site.ts calls it.
    expect(callers).toEqual([
      'extension/panel/lib/allow-site.ts',
      'extension/panel/lib/worker-api.ts',
    ]);
    const lib = readFileSync(path.join(panel, 'lib/allow-site.ts'), 'utf8');
    expect(lib.match(/deps\.allowOrigin\s*\(/g)).toHaveLength(1);
    const card = readFileSync(
      path.join(panel, 'screens/allow-site.tsx'),
      'utf8',
    );
    // The controller's allow() is called once, from the click handler, and
    // that handler is used only as the Allow button's onClick.
    expect(card.match(/\.allow\s*\(/g)).toHaveLength(1);
    expect(card.match(/handleAllow/g)).toHaveLength(2);
    expect(card).toMatch(/onClick=\{handleAllow\}/);
    expect(card).toMatch(
      /const handleAllow = \(\) => \{\s*void controller\.current\?\.allow\(\);/,
    );
    // No raw 'allowOrigin' message anywhere but the worker client.
    const raw = files
      .filter((f) => /['"`]allowOrigin['"`]/.test(readFileSync(f, 'utf8')))
      .map((f) => path.relative(ROOT, f));
    expect(raw).toEqual(['extension/panel/lib/worker-api.ts']);
    // No other screen talks to the worker's allowOrigin.
    const running = readFileSync(
      path.join(panel, 'screens/running.tsx'),
      'utf8',
    );
    expect(running).not.toMatch(/allowOrigin/);
  });
});

describe('the allowed note', () => {
  it('never claims the site is safe', () => {
    const all = [
      ...(Object.values(ALLOW_SITE_COPY) as unknown[]).filter(
        (v): v is string => typeof v === 'string',
      ),
      ALLOW_SITE_COPY.moved(A),
    ].join('\n');
    expect(all).not.toMatch(/\b(safe|secure|trusted|verified)\b/i);
  });
});
