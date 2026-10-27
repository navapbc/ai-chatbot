import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  type Mock,
  vi,
} from 'vitest';
import {
  installPanelBehavior,
  installSweepAlarm,
  onAlarm,
  SAVED_DETAILS_SWEEP_ALARM,
} from '@/extension/src/panel-behavior';

// The worker's panel contract: a toolbar click opens the side panel, and the
// worker owns (but does nothing for) the saved-details sweep alarm.

const alarm = (name: string) =>
  ({ name, scheduledTime: 0 }) as chrome.alarms.Alarm;

const asChrome = (stub: unknown) => stub as typeof chrome;

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('installPanelBehavior', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  it('opens the panel on action click, once', () => {
    const setPanelBehavior = vi.fn(async () => {});
    installPanelBehavior(asChrome({ sidePanel: { setPanelBehavior } }));
    expect(setPanelBehavior).toHaveBeenCalledTimes(1);
    expect(setPanelBehavior).toHaveBeenCalledWith({
      openPanelOnActionClick: true,
    });
    expect(warn).not.toHaveBeenCalled();
  });

  it('swallows a rejection with a value-free warning', async () => {
    const setPanelBehavior = vi.fn(async () => {
      throw new Error('secret-detail');
    });
    expect(() =>
      installPanelBehavior(asChrome({ sidePanel: { setPanelBehavior } })),
    ).not.toThrow();
    await vi.waitFor(() => expect(warn).toHaveBeenCalledTimes(1));
    expect(warn).toHaveBeenCalledWith(
      '[panel] could not set the side panel behavior',
    );
  });

  it('swallows a synchronous throw', () => {
    const setPanelBehavior = vi.fn(() => {
      throw new Error('boom');
    });
    expect(() =>
      installPanelBehavior(asChrome({ sidePanel: { setPanelBehavior } })),
    ).not.toThrow();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('carries on when chrome.sidePanel is missing', () => {
    expect(() => installPanelBehavior(asChrome({}))).not.toThrow();
    expect(warn).toHaveBeenCalledWith(
      '[panel] could not set the side panel behavior',
    );
  });
});

describe('the alarm listener', () => {
  let store: Record<string, unknown>;
  const NOW = Date.now();
  beforeEach(() => {
    store = {};
    vi.stubGlobal('chrome', {
      storage: {
        local: {
          get: vi.fn(async (k: string) =>
            k in store ? { [k]: store[k] } : {},
          ),
          set: vi.fn(),
          remove: vi.fn(async (k: string) => {
            delete store[k];
          }),
        },
      },
    });
  });

  it('deletes an expired record on the sweep alarm', async () => {
    store.savedDetails = {
      savedAt: NOW - 9 * 3_600_000,
      values: { city: 'a' },
    };
    await onAlarm(alarm(SAVED_DETAILS_SWEEP_ALARM));
    expect('savedDetails' in store).toBe(false);
  });

  it('keeps a fresh record', async () => {
    store.savedDetails = { savedAt: NOW, values: { city: 'a' } };
    await onAlarm(alarm(SAVED_DETAILS_SWEEP_ALARM));
    expect('savedDetails' in store).toBe(true);
  });

  it('ignores other alarm names without touching storage', async () => {
    expect(onAlarm(alarm('other'))).toBeUndefined();
    const local = (chrome.storage.local as unknown as { get: Mock }).get;
    expect(local).not.toHaveBeenCalled();
  });

  it('swallows a failing sweep', async () => {
    (chrome.storage.local as unknown as { get: Mock }).get.mockRejectedValue(
      new Error('boom'),
    );
    await expect(
      onAlarm(alarm(SAVED_DETAILS_SWEEP_ALARM)),
    ).resolves.toBeUndefined();
  });
});

describe('installSweepAlarm', () => {
  it('creates the alarm now, on install and on startup, swallowing rejection', async () => {
    const create = vi.fn(async () => {
      throw new Error('boom');
    });
    const onInstalled = { addListener: vi.fn() };
    const onStartup = { addListener: vi.fn() };
    vi.stubGlobal('chrome', { alarms: { create } });
    installSweepAlarm(asChrome({ runtime: { onInstalled, onStartup } }));
    expect(onInstalled.addListener).toHaveBeenCalledTimes(1);
    expect(onStartup.addListener).toHaveBeenCalledTimes(1);
    onInstalled.addListener.mock.calls[0][0]();
    await new Promise((r) => setTimeout(r, 0));
    expect(create).toHaveBeenCalledWith('saved-details-sweep', {
      periodInMinutes: 30,
    });
    expect(create).toHaveBeenCalledTimes(2);
  });
});

describe('installSweepAlarm sweeps at worker start', () => {
  const NOW = Date.now();
  let store: Record<string, unknown>;
  const start = async () => {
    store = {};
    vi.stubGlobal('chrome', {
      alarms: { get: vi.fn(async () => ({})), create: vi.fn() },
      storage: {
        local: {
          get: vi.fn(async (k: string) =>
            k in store ? { [k]: store[k] } : {},
          ),
          remove: vi.fn(async (k: string) => {
            delete store[k];
          }),
        },
      },
    });
  };

  it('deletes an expired record', async () => {
    await start();
    store.savedDetails = {
      savedAt: NOW - 9 * 3_600_000,
      values: { city: 'a' },
    };
    installSweepAlarm(asChrome({ runtime: {} }));
    await vi.waitFor(() => expect('savedDetails' in store).toBe(false));
  });

  it('leaves a fresh record', async () => {
    await start();
    store.savedDetails = { savedAt: NOW, values: { city: 'a' } };
    installSweepAlarm(asChrome({ runtime: {} }));
    await new Promise((r) => setTimeout(r, 10));
    expect('savedDetails' in store).toBe(true);
  });
});

describe('service worker start', () => {
  const listener = () => ({ addListener: vi.fn() });

  async function loadWorker(sidePanel: unknown) {
    vi.resetModules();
    const stub = {
      runtime: {
        id: 'ext',
        onMessage: listener(),
        onInstalled: listener(),
        onStartup: listener(),
      },
      debugger: { onEvent: listener(), onDetach: listener() },
      tabs: { onUpdated: listener(), onRemoved: listener() },
      alarms: { onAlarm: listener(), create: vi.fn(async () => {}) },
      storage: {
        session: {
          get: vi.fn(async () => ({})),
          set: vi.fn(),
          remove: vi.fn(),
        },
        local: { get: vi.fn(async () => ({})), set: vi.fn(), remove: vi.fn() },
      },
      sidePanel,
    };
    vi.stubGlobal('chrome', stub);
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    await import('@/extension/src/service-worker');
    return stub;
  }

  const registered = (stub: Awaited<ReturnType<typeof loadWorker>>) => [
    stub.runtime.onMessage.addListener,
    stub.debugger.onEvent.addListener,
    stub.debugger.onDetach.addListener,
    stub.tabs.onUpdated.addListener,
    stub.tabs.onRemoved.addListener,
    stub.alarms.onAlarm.addListener,
    stub.runtime.onInstalled.addListener,
    stub.runtime.onStartup.addListener,
  ];

  it('sets the panel behavior once and registers every listener', async () => {
    const setPanelBehavior = vi.fn(async () => {});
    const stub = await loadWorker({ setPanelBehavior });
    expect(setPanelBehavior).toHaveBeenCalledTimes(1);
    for (const add of registered(stub)) expect(add).toHaveBeenCalledTimes(1);
  });

  it('still registers every listener when the call rejects or sidePanel is missing', async () => {
    for (const sidePanel of [
      {
        setPanelBehavior: vi.fn(async () => {
          throw new Error('no');
        }),
      },
      undefined,
    ]) {
      const stub = await loadWorker(sidePanel);
      for (const add of registered(stub)) expect(add).toHaveBeenCalledTimes(1);
    }
  });
});
