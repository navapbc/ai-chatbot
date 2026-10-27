import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  installPanelBehavior,
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
  it('ignores other alarm names and does nothing for the sweep', () => {
    expect(onAlarm(alarm('other'))).toBeUndefined();
    expect(onAlarm(alarm(SAVED_DETAILS_SWEEP_ALARM))).toBeUndefined();
  });
});

describe('service worker start', () => {
  const listener = () => ({ addListener: vi.fn() });

  async function loadWorker(sidePanel: unknown) {
    vi.resetModules();
    const stub = {
      runtime: { id: 'ext', onMessage: listener() },
      debugger: { onEvent: listener(), onDetach: listener() },
      tabs: { onUpdated: listener(), onRemoved: listener() },
      alarms: { onAlarm: listener() },
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
