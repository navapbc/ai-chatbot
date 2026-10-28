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
      <h2 className="text-base font-semibold">Home</h2>
      {sites.length > 0 ? (
        <div className="flex flex-col gap-2">
          {sites.map((site) => (
            <Button
              key={site.id}
              type="button"
              onClick={() => handleSite(site)}
            >
              {site.name}
            </Button>
          ))}
        </div>
      ) : null}
      <Button
        type="button"
        variant={sites.length > 0 ? 'outline' : 'default'}
        onClick={handleUseThisTab}
      >
        Use this tab
      </Button>
      {message ? (
        <p role="alert" className="text-sm">
          {message}
        </p>
      ) : null}
      {recents.length > 0 ? (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold">Recent</h3>
          <ul className="flex flex-col gap-1 text-sm" aria-label="Recent runs">
            {recents.map((r) => (
              <li key={r.runId}>
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
