import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { RelaySite } from '../../../lib/relay/config';
import { normalizeOrigin } from '../../../lib/relay/policy';
import { HOME_COPY } from '../lib/copy';
import {
  formatAgo,
  loadRecents,
  type Recent,
  visibleRecents,
} from '../lib/recents';
import type { Target } from '../lib/state';

type Props = {
  sites: RelaySite[];
  onChoose: (target: Target) => void;
};

export const HomeScreen = ({ sites, onChoose }: Props) => {
  const [recents, setRecents] = useState<Recent[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [tabLabel, setTabLabel] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    chrome.tabs
      .query({ active: true, lastFocusedWindow: true })
      .then(([tab]) => {
        if (!current || !tab?.url) return;
        try {
          const url = new URL(tab.url);
          if (url.protocol === 'https:')
            setTabLabel(url.hostname + url.pathname);
        } catch {}
      })
      .catch(() => {});
    return () => {
      current = false;
    };
  }, []);

  useEffect(() => {
    let current = true;
    loadRecents().then((list) => current && setRecents(visibleRecents(list)));
    return () => {
      current = false;
    };
  }, []);

  const handleSite = async (site: RelaySite) => {
    setMessage(null);
    const origin = normalizeOrigin(site.url);
    if (!origin) {
      setMessage(HOME_COPY.cannotOpen);
      return;
    }
    try {
      const tab = await chrome.tabs.create({ url: site.url, active: true });
      if (tab?.id === undefined) {
        setMessage(HOME_COPY.cannotOpen);
        return;
      }
      onChoose({ tabId: tab.id, origin, siteName: site.name });
    } catch {
      setMessage(HOME_COPY.cannotOpen);
    }
  };

  const handleUseThisTab = async () => {
    setMessage(null);
    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        lastFocusedWindow: true,
      });
      const origin = tab?.url ? normalizeOrigin(tab.url) : null;
      if (tab?.id === undefined || !tab.url || !origin) {
        setMessage(HOME_COPY.httpsOnly);
        return;
      }
      onChoose({
        tabId: tab.id,
        origin,
        siteName: new URL(tab.url).hostname,
      });
    } catch {
      setMessage(HOME_COPY.httpsOnly);
    }
  };

  const now = Date.now();

  return (
    <section className="flex flex-col gap-4" aria-label="Home">
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold">Start a form</h2>
        <p className="text-sm text-muted-foreground">
          Where should the agent work?
        </p>
      </div>
      <div className="flex flex-col gap-3 rounded-xl bg-accent p-3">
        <div className="flex flex-col gap-0.5 text-sm">
          <span className="font-semibold">This site</span>
          <span className="truncate text-muted-foreground">
            {tabLabel ?? 'The tab you are on'}
          </span>
        </div>
        <Button type="button" onClick={handleUseThisTab}>
          Use this tab
        </Button>
      </div>
      {message ? (
        <p role="alert" className="text-sm">
          {message}
        </p>
      ) : null}
      {sites.length > 0 ? (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm text-muted-foreground">
            Or go to a common benefits site
          </h3>
          <ul className="flex flex-col gap-2" aria-label="Benefits sites">
            {sites.map((site) => (
              <li key={site.id}>
                <button
                  type="button"
                  onClick={() => handleSite(site)}
                  className="flex w-full items-center gap-3 rounded-xl border border-border bg-secondary p-3 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <span
                    aria-hidden="true"
                    className="flex size-7 shrink-0 items-center justify-center rounded-md bg-accent text-xs font-semibold text-primary"
                  >
                    {site.name.slice(0, 2).toUpperCase()}
                  </span>
                  <span className="flex min-w-0 flex-col">
                    <span className="font-semibold">{site.name}</span>
                    <span className="truncate text-muted-foreground">
                      {new URL(site.url).hostname}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {recents.length > 0 ? (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold">Recent runs</h3>
          <ul className="flex flex-col gap-1 text-sm" aria-label="Recent runs">
            {recents.map((r) => (
              <li
                key={r.runId}
                className="rounded-md border border-border px-3 py-2"
              >
                {r.siteName}, {formatAgo(now, r.startedAt)}
                {r.endedReason ? HOME_COPY.ended : ''}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
};
