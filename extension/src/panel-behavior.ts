/// <reference types="chrome" />

import {
  ensureSweepAlarm,
  SAVED_DETAILS_SWEEP_ALARM,
  sweepSavedDetails,
} from '../panel/lib/saved-details';

export { SAVED_DETAILS_SWEEP_ALARM };

/**
 * Makes a click on the toolbar icon open the side panel. Never throws or
 * rejects: an older Chrome without `sidePanel`, or a refused call, only logs
 * a value-free warning so worker start and the other listeners carry on.
 */
export function installPanelBehavior(chromeApi: typeof chrome): void {
  const warn = () =>
    console.warn('[panel] could not set the side panel behavior');
  try {
    const result = chromeApi.sidePanel?.setPanelBehavior({
      openPanelOnActionClick: true,
    });
    if (!result) {
      warn();
      return;
    }
    Promise.resolve(result).catch(warn);
  } catch {
    warn();
  }
}

/**
 * The worker runs the saved-details sweep, so an expired record is deleted
 * even with the panel closed. Every other alarm is ignored. Never rejects.
 */
export function onAlarm(alarm: chrome.alarms.Alarm): Promise<void> | undefined {
  if (alarm.name !== SAVED_DETAILS_SWEEP_ALARM) return undefined;
  return sweepSavedDetails().catch(() => {});
}

/** Creates the sweep alarm on install, browser start and worker start. */
export function installSweepAlarm(chromeApi: typeof chrome): void {
  const create = () => void ensureSweepAlarm();
  chromeApi.runtime.onInstalled?.addListener(create);
  chromeApi.runtime.onStartup?.addListener(create);
  create();
}
