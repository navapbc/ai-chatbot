import { describe, expect, it } from 'vitest';
import {
  ALLOWED_EVENT_METHODS,
  checkCommand,
  filterEvent,
  inScope,
  isJavascriptUrl,
  isMethodAllowed,
  normalizeOrigin,
} from '@/lib/relay/policy';

const ctx = { origin: 'https://a.gov' };

describe('normalizeOrigin', () => {
  it('returns the origin for https URLs', () => {
    expect(normalizeOrigin('https://a.gov/x?y')).toBe('https://a.gov');
    expect(normalizeOrigin('https://a.gov:8443/x')).toBe('https://a.gov:8443');
  });

  it('returns null for non-https and unparseable input', () => {
    for (const url of [
      'http://a.gov',
      'about:blank',
      'chrome://x',
      'not a url',
      '',
    ]) {
      expect(normalizeOrigin(url)).toBeNull();
    }
  });
});

describe('inScope', () => {
  it('matches the exact origin', () => {
    expect(inScope('https://a.gov/page', 'https://a.gov')).toBe(true);
  });

  it('rejects a different port, subdomain, lookalike, scheme or junk', () => {
    expect(inScope('https://a.gov:8443/', 'https://a.gov')).toBe(false);
    expect(inScope('https://sub.a.gov/', 'https://a.gov')).toBe(false);
    expect(inScope('https://a.gov.evil.com/', 'https://a.gov')).toBe(false);
    expect(inScope('https://evil.com/?https://a.gov', 'https://a.gov')).toBe(
      false,
    );
    expect(inScope('http://a.gov/', 'https://a.gov')).toBe(false);
    expect(inScope('about:blank', 'https://a.gov')).toBe(false);
    expect(inScope('', 'https://a.gov')).toBe(false);
  });
});

describe('isMethodAllowed', () => {
  it('allows listed methods and refuses unknown ones', () => {
    expect(isMethodAllowed('Runtime.evaluate')).toBe(true);
    expect(isMethodAllowed('Network.getCookies')).toBe(false);
    expect(isMethodAllowed('')).toBe(false);
  });
});

describe('checkCommand', () => {
  it('refuses cookie, storage, fetch and download methods', () => {
    for (const method of [
      'Network.getAllCookies',
      'Network.getCookies',
      'Storage.getCookies',
      'Fetch.enable',
      'Page.setDownloadBehavior',
      'Not.AMethod',
    ]) {
      expect(checkCommand({ method }, ctx).ok).toBe(false);
    }
  });

  it('allows ordinary page-driving methods', () => {
    expect(
      checkCommand({ method: 'Accessibility.getFullAXTree' }, ctx),
    ).toEqual({ ok: true });
    expect(
      checkCommand({ method: 'Input.insertText', params: { text: 'x' } }, ctx),
    ).toEqual({ ok: true });
  });

  it('checks Page.navigate against scope', () => {
    expect(
      checkCommand(
        { method: 'Page.navigate', params: { url: 'https://a.gov/next' } },
        ctx,
      ),
    ).toEqual({ ok: true });
    const out = checkCommand(
      { method: 'Page.navigate', params: { url: 'https://b.gov/' } },
      ctx,
    );
    expect(out.ok).toBe(false);
  });

  it('checks Target.createTarget against scope, including a missing url', () => {
    expect(
      checkCommand(
        { method: 'Target.createTarget', params: { url: 'https://a.gov/' } },
        ctx,
      ),
    ).toEqual({ ok: true });
    expect(
      checkCommand(
        { method: 'Target.createTarget', params: { url: 'https://b.gov/' } },
        ctx,
      ).ok,
    ).toBe(false);
    expect(checkCommand({ method: 'Target.createTarget' }, ctx).ok).toBe(false);
    expect(
      checkCommand({ method: 'Target.createTarget', params: {} }, ctx).ok,
    ).toBe(false);
  });

  it('refuses malformed params without throwing', () => {
    for (const params of [undefined, null, 'x', 5, [], {}, { url: 5 }]) {
      expect(checkCommand({ method: 'Page.navigate', params }, ctx).ok).toBe(
        false,
      );
    }
  });
});

const URL_METHODS = ['Page.navigate', 'Target.createTarget'];
const BAD_PARAMS: unknown[] = [
  undefined,
  null,
  'x',
  5,
  [],
  {},
  { url: 5 },
  { url: {} },
];

describe('command policy regressions', () => {
  it('normalizes https://a.gov@evil.com to evil.com, out of scope for a.gov', () => {
    expect(normalizeOrigin('https://a.gov@evil.com')).toBe('https://evil.com');
    expect(inScope('https://a.gov@evil.com/x', 'https://a.gov')).toBe(false);
    for (const method of URL_METHODS) {
      expect(
        checkCommand({ method, params: { url: 'https://a.gov@evil.com' } }, ctx)
          .ok,
      ).toBe(false);
    }
  });

  it('refuses javascript:, data:, blob: and file: urls for navigation and new targets', () => {
    for (const method of URL_METHODS) {
      for (const url of [
        'javascript:alert(1)',
        'JaVaScRiPt:alert(1)',
        'data:text/html,<p>x</p>',
        'blob:https://a.gov/uuid',
        'file:///etc/passwd',
      ]) {
        expect(checkCommand({ method, params: { url } }, ctx).ok).toBe(false);
      }
    }
  });

  it('refuses lowercase method names', () => {
    for (const method of [
      'page.navigate',
      'runtime.evaluate',
      'PAGE.NAVIGATE',
    ]) {
      expect(isMethodAllowed(method)).toBe(false);
      expect(
        checkCommand({ method, params: { url: 'https://a.gov/' } }, ctx).ok,
      ).toBe(false);
    }
  });

  it('refuses non-string and non-object params for every URL-checked method', () => {
    for (const method of URL_METHODS) {
      for (const params of BAD_PARAMS) {
        expect(checkCommand({ method, params }, ctx).ok).toBe(false);
      }
    }
  });

  it('refuses non-object params for methods that require them', () => {
    for (const method of [
      'Runtime.evaluate',
      'Input.insertText',
      'Runtime.callFunctionOn',
    ]) {
      for (const params of ['x', 5, [], null]) {
        expect(checkCommand({ method, params }, ctx).ok).toBe(false);
      }
    }
  });

  it('refuses Page.disable: the bridge owns the Page domain', () => {
    expect(isMethodAllowed('Page.disable')).toBe(false);
    expect(checkCommand({ method: 'Page.disable' }, ctx).ok).toBe(false);
    expect(isMethodAllowed('Page.enable')).toBe(true);
  });
});

// Events the agent receives while driving every documented browser command,
// measured by tests/relay/e2e/allowlist.e2e.test.ts (ALLOWLIST_VERBOSE=1).
const MEASURED_EVENTS = [
  'Accessibility.loadComplete',
  'DOM.documentUpdated',
  'Network.loadingFinished',
  'Network.requestWillBeSent',
  'Network.responseReceived',
  'Page.backForwardCacheNotUsed',
  'Page.domContentEventFired',
  'Page.frameAttached',
  'Page.frameDetached',
  'Page.frameNavigated',
  'Page.frameRequestedNavigation',
  'Page.frameStartedLoading',
  'Page.frameStartedNavigating',
  'Page.frameStoppedLoading',
  'Page.javascriptDialogClosed',
  'Page.javascriptDialogOpening',
  'Page.loadEventFired',
  'Runtime.executionContextCreated',
  'Runtime.executionContextDestroyed',
  'Runtime.executionContextsCleared',
  'Target.attachedToTarget',
  'Target.detachedFromTarget',
  'Target.targetInfoChanged',
];

// Allowed events that the measured run did not show, each with its reason.
const UNMEASURED_JUSTIFICATIONS: Record<string, string> = {
  'Network.loadingFailed': 'idle waits count failed requests (existing set)',
  'Network.requestServedFromCache': 'idle waits count cached requests',
  'Page.navigatedWithinDocument': 'history.pushState navigations settle waits',
};

describe('event allowlist', () => {
  it('an unlisted event is dropped', () => {
    for (const method of [
      'Page.compilationCacheProduced',
      'DOM.attributeModified',
      'DOM.characterDataModified',
      'Accessibility.nodesUpdated',
      'Log.entryAdded',
      'Console.messageAdded',
      'Made.Up',
      'page.frameNavigated',
    ]) {
      expect(filterEvent(method, { x: 1 })).toBeNull();
    }
  });

  it('every allowed event is in the measured list or has a justification entry', () => {
    for (const method of ALLOWED_EVENT_METHODS) {
      expect(
        MEASURED_EVENTS.includes(method) || method in UNMEASURED_JUSTIFICATIONS,
        method,
      ).toBe(true);
    }
    for (const method of Object.keys(UNMEASURED_JUSTIFICATIONS)) {
      expect(ALLOWED_EVENT_METHODS.has(method), method).toBe(true);
      expect(MEASURED_EVENTS.includes(method), method).toBe(false);
    }
  });

  it('console and exception events are dropped', () => {
    for (const method of [
      'Runtime.consoleAPICalled',
      'Runtime.exceptionThrown',
      'Runtime.bindingCalled',
      'Runtime.inspectRequested',
      'Runtime.exceptionRevoked',
      'Log.entryAdded',
    ]) {
      expect(ALLOWED_EVENT_METHODS.has(method)).toBe(false);
      expect(filterEvent(method, { text: 'secret' })).toBeNull();
    }
  });

  it('keeps the events agent-browser waits on', () => {
    for (const method of [
      'Page.frameNavigated',
      'Page.loadEventFired',
      'Runtime.executionContextCreated',
      'Target.attachedToTarget',
    ]) {
      expect(filterEvent(method, {})).not.toBeNull();
    }
  });
});

describe('filterEvent', () => {
  const COOKIE = 'sid=cookie-sentinel-1';
  const AUTH = 'Bearer auth-sentinel-2';
  const SET_COOKIE = 'sid=set-cookie-sentinel-3; HttpOnly';
  const BODY = 'post-body-sentinel-4';
  const SECRETS = [
    COOKIE,
    AUTH,
    SET_COOKIE,
    BODY,
    'cookie-sentinel',
    'auth-sentinel',
  ];

  const headers = {
    Cookie: COOKIE,
    Authorization: AUTH,
    'Set-Cookie': SET_COOKIE,
  };
  const headersText = `GET / HTTP/1.1\r\nCookie: ${COOKIE}\r\nAuthorization: ${AUTH}`;

  const request = {
    url: 'https://a.gov/api',
    method: 'POST',
    headers,
    postData: BODY,
    hasPostData: true,
    postDataEntries: [{ bytes: BODY }],
  };
  const response = {
    url: 'https://a.gov/api',
    status: 200,
    statusText: 'OK',
    headers,
    headersText,
    requestHeaders: headers,
    requestHeadersText: headersText,
    mimeType: 'application/json',
  };

  const leaks = (value: unknown) => {
    const text = JSON.stringify(value);
    return SECRETS.filter((s) => text.includes(s));
  };

  it('drops every ExtraInfo and early-hints Network event', () => {
    for (const method of [
      'Network.requestWillBeSentExtraInfo',
      'Network.responseReceivedExtraInfo',
      'Network.responseReceivedEarlyHints',
      'Network.webTransportCreatedExtraInfo',
      'Network.somethingNewExtraInfo',
    ]) {
      expect(
        filterEvent(method, { requestId: '1', headers, associatedCookies: [] }),
      ).toBeNull();
    }
  });

  it('drops Network events outside the forwarded set', () => {
    for (const method of [
      'Network.dataReceived',
      'Network.webSocketFrameSent',
      'Network.webSocketFrameReceived',
      'Network.eventSourceMessageReceived',
      'Network.reportingApiReportAdded',
    ]) {
      expect(filterEvent(method, { requestId: '1', response })).toBeNull();
    }
  });

  it('strips headers, cookies and post data from forwarded Network events', () => {
    const events: Array<[string, unknown]> = [
      [
        'Network.requestWillBeSent',
        {
          requestId: '1',
          loaderId: 'L',
          documentURL: 'https://a.gov/',
          request,
          timestamp: 1,
          wallTime: 2,
          type: 'Fetch',
          redirectResponse: response,
          initiator: { type: 'script' },
        },
      ],
      [
        'Network.responseReceived',
        {
          requestId: '1',
          loaderId: 'L',
          timestamp: 3,
          type: 'Fetch',
          response,
          hasExtraInfo: true,
        },
      ],
      [
        'Network.loadingFinished',
        { requestId: '1', timestamp: 4, encodedDataLength: 10 },
      ],
      [
        'Network.loadingFailed',
        {
          requestId: '1',
          timestamp: 5,
          type: 'Fetch',
          errorText: 'net::ERR_FAILED',
          blockedCookies: [{ cookie: { value: COOKIE } }],
          cookiePartitionKey: COOKIE,
          associatedCookies: [{ cookie: { name: 'sid', value: COOKIE } }],
        },
      ],
      ['Network.requestServedFromCache', { requestId: '1' }],
    ];
    for (const [method, params] of events) {
      const out = filterEvent(method, params);
      expect(out, method).not.toBeNull();
      expect(out?.method).toBe(method);
      expect(leaks(out), method).toEqual([]);
    }
  });

  it('keeps the fields agent-browser waits on', () => {
    const sent = filterEvent('Network.requestWillBeSent', {
      requestId: '1',
      loaderId: 'L',
      request,
      timestamp: 1,
      type: 'Fetch',
    });
    expect(sent?.params).toEqual({
      requestId: '1',
      loaderId: 'L',
      request: { url: 'https://a.gov/api', method: 'POST' },
      timestamp: 1,
      type: 'Fetch',
    });

    const received = filterEvent('Network.responseReceived', {
      requestId: '1',
      timestamp: 3,
      type: 'Fetch',
      response,
    });
    expect(received?.params).toEqual({
      requestId: '1',
      timestamp: 3,
      type: 'Fetch',
      response: {
        url: 'https://a.gov/api',
        status: 200,
        statusText: 'OK',
        mimeType: 'application/json',
      },
    });

    const failed = filterEvent('Network.loadingFailed', {
      requestId: '1',
      errorText: 'net::ERR_FAILED',
    });
    expect(failed?.params).toEqual({
      requestId: '1',
      errorText: 'net::ERR_FAILED',
    });
  });

  it('passes other domains through unchanged', () => {
    const params = { frameId: 'F', url: 'https://a.gov/', nested: { a: [1] } };
    expect(filterEvent('Page.frameNavigated', params)).toEqual({
      method: 'Page.frameNavigated',
      params,
    });
    expect(filterEvent('Target.attachedToTarget', params)?.params).toBe(params);
  });

  it('does not throw on malformed Network params', () => {
    for (const params of [undefined, null, 'x', 5, [], { request: null }]) {
      expect(() =>
        filterEvent('Network.requestWillBeSent', params),
      ).not.toThrow();
    }
  });
});

describe('isJavascriptUrl', () => {
  it('recognizes javascript: URLs the way a browser parses them', () => {
    for (const url of [
      'javascript:void(0)',
      'JavaScript:void(0)',
      'JAVASCRIPT:x',
      'javascript:',
      '\tjavascript:x',
      '\n javascript:x',
      ' javascript:__doPostBack("a","")',
      'java\tscript:x',
    ]) {
      expect(isJavascriptUrl(url), JSON.stringify(url)).toBe(true);
    }
  });

  it('treats every other or malformed URL as not ignorable', () => {
    for (const url of [
      'about:blank',
      'data:text/html,x',
      'blob:https://a.gov/1',
      'https://a.gov/',
      'http://a.gov/',
      'javascript',
      'xjavascript:x',
      '',
      '::not a url',
      'http://[bad',
    ]) {
      expect(isJavascriptUrl(url), JSON.stringify(url)).toBe(false);
    }
  });
});
