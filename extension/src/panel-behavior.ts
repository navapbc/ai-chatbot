/// <reference types="chrome" />

/** The alarm the panel owns: it sweeps expired saved details (Task 6). */
export const SAVED_DETAILS_SWEEP_ALARM = 'saved-details-sweep';

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
 * The sweep itself runs in the panel; the worker listens only so the owner
 * of the alarm is explicit. Every alarm is ignored here.
 */
export function onAlarm(alarm: chrome.alarms.Alarm): void {
  if (alarm.name !== SAVED_DETAILS_SWEEP_ALARM) return;
}
