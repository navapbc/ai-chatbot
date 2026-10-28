import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  type AllowSiteController,
  type AllowSiteDeps,
  type AllowSiteNotice,
  type AllowSiteView,
  createAllowSiteController,
} from '../lib/allow-site';
import { ALLOW_SITE_COPY } from '../lib/copy';
import { createWorkerApi } from '../lib/worker-api';

// The allow-this-site card: the worker holds the run because the tab went to
// another https origin, and only the caseworker decides whether the agent may
// follow it, for this run. A thin view over createAllowSiteController (see
// there for the guarantees): the Allow button's click is the only thing that
// calls the controller's allow(). The parent keys the card by the pending
// origin, so a new origin gets a new card (and a new controller) naming it.
//
// The origin is the worker's validated https origin and the tab line is the
// tab's origin plus path: both render as text only, and the tab line lives
// only in this component's state (never logged, never stored).

export type AllowSiteCardDeps = Omit<AllowSiteDeps, 'onAllowed'>;

/** Chrome's tab API and the real worker client. */
export const defaultAllowSiteDeps = (): AllowSiteCardDeps => ({
  allowOrigin: createWorkerApi().allowOrigin,
  getTab: (tabId) => chrome.tabs.get(tabId),
  now: () => Date.now(),
  setInterval: (fn, ms) => setInterval(fn, ms),
  clearInterval: (id) => clearInterval(id),
});

const NOTICE_TEXT: Record<AllowSiteNotice, string> = {
  holdEnded: ALLOW_SITE_COPY.holdEnded,
  tooMany: ALLOW_SITE_COPY.tooMany,
  unreachable: ALLOW_SITE_COPY.unreachable,
};

type Props = {
  /** The pending origin this card asks about. */
  origin: string;
  /** The session's current pending origin (undefined once it cleared). */
  pendingOrigin: string | undefined;
  stopping: boolean;
  tabId: number;
  /** From the session; absent when the hold began before the panel saw it. */
  holdEndsAt?: number;
  deps: AllowSiteCardDeps;
  onAllowed: () => void;
  onStop: () => void;
};

export const AllowSiteCard = ({
  origin,
  pendingOrigin,
  stopping,
  tabId,
  holdEndsAt,
  deps,
  onAllowed,
  onStop,
}: Props) => {
  const controller = useRef<AllowSiteController | null>(null);
  const [view, setView] = useState<AllowSiteView | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const noticeRef = useRef<HTMLParagraphElement>(null);
  // Read at mount only: the card is keyed by origin, and these do not change
  // for one hold.
  const mountInput = useRef({ origin, tabId, holdEndsAt, deps, onAllowed });

  useEffect(() => {
    const input = mountInput.current;
    const current = createAllowSiteController(
      {
        origin: input.origin,
        tabId: input.tabId,
        holdEndsAt: input.holdEndsAt,
      },
      { ...input.deps, onAllowed: () => input.onAllowed() },
      setView,
    );
    controller.current = current;
    current.start();
    return () => {
      current.dispose();
      controller.current = null;
    };
  }, []);

  useEffect(() => {
    controller.current?.update({ pendingOrigin, stopping });
  }, [pendingOrigin, stopping]);

  // Once, when the card appears: not on every poll.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  const phase = view?.phase ?? 'waiting';
  const notice = view?.notice ?? null;
  // Allow may have disabled itself: keep keyboard focus on what explains it.
  useEffect(() => {
    if (notice) noticeRef.current?.focus();
  }, [notice]);

  const handleAllow = () => {
    void controller.current?.allow();
  };
  const handleStop = () => onStop();

  const secondsLeft = view?.secondsLeft ?? null;
  const canAllow = view?.canAllow === true;
  const allowing = phase === 'allowing';

  return (
    <section
      aria-labelledby="allow-site-title"
      className="flex flex-col gap-2 rounded-md border border-border p-3"
    >
      <h3
        id="allow-site-title"
        ref={headingRef}
        tabIndex={-1}
        className="text-sm font-semibold"
      >
        {ALLOW_SITE_COPY.title}
      </h3>
      <p role="alert" className="text-sm">
        {ALLOW_SITE_COPY.moved(origin)}
      </p>
      {view?.tabText ? (
        <p className="text-sm break-all">
          {ALLOW_SITE_COPY.tabNow(view.tabText)}
        </p>
      ) : null}
      <p className="text-sm">{ALLOW_SITE_COPY.hint}</p>
      {phase === 'waiting' || phase === 'allowing' ? (
        // Informational: the worker decides whether the hold is still on.
        <p className="text-sm">
          {secondsLeft === null
            ? ALLOW_SITE_COPY.decideSoon
            : secondsLeft > 0
              ? ALLOW_SITE_COPY.secondsLeft(secondsLeft)
              : ALLOW_SITE_COPY.holdMayHaveEnded}
        </p>
      ) : null}
      {notice ? (
        <p ref={noticeRef} role="alert" tabIndex={-1} className="text-sm">
          {NOTICE_TEXT[notice]}
        </p>
      ) : null}
      <div className="flex flex-col gap-2">
        <Button
          type="button"
          onClick={handleAllow}
          disabled={!canAllow}
          aria-busy={allowing}
        >
          {allowing ? ALLOW_SITE_COPY.allowing : ALLOW_SITE_COPY.allow}
        </Button>
        <Button type="button" variant="outline" onClick={handleStop}>
          {ALLOW_SITE_COPY.stop}
        </Button>
      </div>
    </section>
  );
};
