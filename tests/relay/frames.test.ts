import { describe, expect, it } from 'vitest';
import { ALLOWED_CDP_METHODS } from '@/lib/relay/allowed-cdp-methods';
import {
  commandAddresses,
  contextIdFromObjectId,
  contextInScope,
  createFrameScope,
  documentOrigin,
  destinationAllowed,
  type FrameInfo,
  frameInScope,
  framesOfTree,
  inRunOrigins,
  NO_FRAME_CONTENT_METHODS,
  navigationDestination,
  PRUNED_RESULT_METHODS,
  pruneAXTree,
  pruneDescribedNode,
  pruneFrameTree,
  scopeFrameEvent,
} from '@/lib/relay/frames';

const RUN = 'https://a.gov';
const OTHER = 'https://b.a.gov';
const origins = { origin: RUN, allowedFrameOrigins: [] as string[] };

const frames = new Map<string, FrameInfo>([
  ['main', { url: `${RUN}/form` }],
  ['srcdoc', { url: 'about:srcdoc', parentId: 'main' }],
  ['blank', { url: 'about:blank', parentId: 'main' }],
  ['fresh', { url: '', parentId: 'main' }],
  ['same', { url: `${RUN}/inner`, parentId: 'main' }],
  ['other', { url: `${OTHER}/frame`, parentId: 'main' }],
  ['other-srcdoc', { url: 'about:srcdoc', parentId: 'other' }],
  ['other-blank-blank', { url: 'about:blank', parentId: 'other-srcdoc' }],
  ['same-in-other', { url: `${RUN}/x`, parentId: 'other' }],
  ['orphan', { url: 'about:blank', parentId: 'gone' }],
  ['root-blank', { url: 'about:blank' }],
  ['userinfo', { url: `https://user:pw@a.gov/x`, parentId: 'main' }],
  ['userinfo-host', { url: 'https://a.gov@evil.example/x', parentId: 'main' }],
  ['data', { url: 'data:text/html,hi', parentId: 'main' }],
  ['blob', { url: `blob:${RUN}/6f1c`, parentId: 'main' }],
  ['http', { url: 'http://a.gov/x', parentId: 'main' }],
  ['upper-blank', { url: 'ABOUT:BLANK', parentId: 'main' }],
  ['blank-hash', { url: 'about:blank#top', parentId: 'main' }],
  ['loop-a', { url: 'about:blank', parentId: 'loop-b' }],
  ['loop-b', { url: 'about:srcdoc', parentId: 'loop-a' }],
]);

describe('frameInScope', () => {
  it.each([
    ['main', true, 'the run origin'],
    ['same', true, 'a same-origin iframe'],
    ['srcdoc', true, 'a srcdoc iframe inherits its parent'],
    ['blank', true, 'about:blank inherits its parent'],
    ['fresh', true, 'a frame not yet navigated holds its initial about:blank'],
    ['blank-hash', true, 'about:blank with a fragment still inherits'],
    ['userinfo', true, 'userinfo does not change the origin'],
    ['other', false, 'a same-site but cross-origin frame'],
    ['other-srcdoc', false, 'srcdoc under an out-of-scope parent'],
    ['other-blank-blank', false, 'the whole parent chain is followed'],
    ['same-in-other', true, 'a run-origin frame nested in another origin'],
    ['orphan', false, 'about:blank whose parent is unknown'],
    ['root-blank', false, 'about:blank with no parent at all'],
    ['userinfo-host', false, 'a userinfo trick naming another host'],
    ['data', false, 'data: has an opaque origin'],
    ['blob', false, 'blob: is not an https origin (fails closed)'],
    ['http', false, 'plain http'],
    ['upper-blank', false, 'a non-canonical about: URL'],
    ['loop-a', false, 'a parent cycle'],
    ['unknown', false, 'an unknown frame'],
  ])('%s -> %s (%s)', (frameId, expected) => {
    expect(frameInScope(frames, frameId, RUN, [])).toBe(expected);
  });

  it('admits an allowed frame origin and the frames that inherit from it', () => {
    for (const id of ['other', 'other-srcdoc', 'other-blank-blank']) {
      expect(frameInScope(frames, id, RUN, [OTHER])).toBe(true);
    }
    expect(frameInScope(frames, 'data', RUN, [OTHER])).toBe(false);
  });

  it('admits an extra run origin (allowOrigin) like the run origin', () => {
    for (const id of ['other', 'other-srcdoc', 'other-blank-blank']) {
      expect(frameInScope(frames, id, RUN, [], [OTHER])).toBe(true);
    }
    expect(frameInScope(frames, 'data', RUN, [], [OTHER])).toBe(false);
    expect(frameInScope(frames, 'http', RUN, [], [OTHER])).toBe(false);
  });
});

describe('inRunOrigins', () => {
  it('is the run origin or an extra run origin, never a frame origin', () => {
    const o = {
      origin: RUN,
      allowedFrameOrigins: ['https://frames.gov'],
      extraOrigins: [OTHER],
    };
    expect(inRunOrigins(`${RUN}/x?y=1`, o)).toBe(true);
    expect(inRunOrigins(`${OTHER}/login`, o)).toBe(true);
    expect(inRunOrigins('https://frames.gov/x', o)).toBe(false);
    expect(inRunOrigins('http://b.a.gov/login', o)).toBe(false);
    expect(inRunOrigins('about:blank', o)).toBe(false);
    expect(inRunOrigins(`${OTHER}/x`, { origin: RUN })).toBe(false);
  });

  it('contextInScope and scopeFrameEvent honour extra origins', () => {
    const scope = createFrameScope();
    const withExtra = { ...origins, extraOrigins: [OTHER] };
    scope.frames.set('main', { url: `${OTHER}/login` });
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextCreated', {
        context: { id: 9, origin: OTHER, auxData: { frameId: 'main' } },
      }),
    ).toBe(false);
    expect(contextInScope(scope, origins, 9)).toBe(false);
    expect(contextInScope(scope, withExtra, 9)).toBe(true);
    expect(
      scopeFrameEvent(scope, withExtra, 'Page.frameNavigated', {
        frame: { id: 'main', url: `${OTHER}/next` },
      }),
    ).toBe(true);
  });
});

describe('contextIdFromObjectId', () => {
  it('reads the middle field of an isolate.context.object id', () => {
    expect(contextIdFromObjectId('-707174053783962030.5.1')).toBe(5);
    expect(contextIdFromObjectId('8807932963341990558.11.2')).toBe(11);
  });

  it('is undefined for anything else', () => {
    for (const id of [
      '',
      '5',
      '1.2',
      '1.2.3.4',
      '{"injectedScriptId":3,"id":1}',
      '1.x.2',
      '1.-2.3',
    ]) {
      expect(contextIdFromObjectId(id)).toBeUndefined();
    }
  });
});

describe('commandAddresses', () => {
  it('collects frame, context, object and node handles', () => {
    expect(
      commandAddresses({
        frameId: 'f',
        executionContextId: 3,
        contextId: 4,
        uniqueContextId: 'u',
        objectId: '1.5.1',
        arguments: [{ objectId: '1.6.1' }, { value: 2 }],
        backendNodeId: 9,
      }),
    ).toEqual({
      frameIds: ['f'],
      contextIds: [3, 4],
      uniqueContextIds: ['u'],
      objectIds: ['1.5.1', '1.6.1'],
      nodes: [{ backendNodeId: 9 }],
    });
  });

  it('is empty for params with no handles', () => {
    expect(commandAddresses({ expression: '1' })).toEqual({
      frameIds: [],
      contextIds: [],
      uniqueContextIds: [],
      objectIds: [],
      nodes: [],
    });
    expect(commandAddresses(undefined).frameIds).toEqual([]);
  });

  it('keeps malformed handles so the caller refuses them', () => {
    const a = commandAddresses({ frameId: 7, contextId: 'x', nodeId: '1' });
    expect(a.frameIds).toEqual([7]);
    expect(a.contextIds).toEqual(['x']);
    expect(a.nodes).toEqual([{ nodeId: '1' }]);
  });
});

const keepOnly =
  (...ids: string[]) =>
  (id: string) =>
    ids.includes(id);

describe('pruneAXTree', () => {
  it('drops out-of-scope frame nodes, their descendants and references', () => {
    const tree = {
      nodes: [
        { nodeId: '1', childIds: ['2', '3'] },
        { nodeId: '2', parentId: '1', frameId: 'main' },
        { nodeId: '3', parentId: '1', frameId: 'other', childIds: ['4'] },
        { nodeId: '4', parentId: '3', childIds: ['5'] },
        { nodeId: '5', parentId: '4', name: { value: 'secret' } },
        { nodeId: '6', parentId: '5' },
      ],
    };
    const out = pruneAXTree(tree, keepOnly('main')) as typeof tree;
    expect(out.nodes.map((n) => n.nodeId)).toEqual(['1', '2']);
    expect(out.nodes[0].childIds).toEqual(['2']);
    expect(JSON.stringify(out)).not.toContain('secret');
  });

  it('drops a node whose frameId is not a string', () => {
    const out = pruneAXTree(
      { nodes: [{ nodeId: '1' }, { nodeId: '2', frameId: 3 }] },
      () => true,
    ) as { nodes: unknown[] };
    expect(out.nodes).toHaveLength(1);
  });

  it('passes a result without nodes through', () => {
    expect(pruneAXTree({ x: 1 }, () => false)).toEqual({ x: 1 });
  });
});

const tree = {
  frameTree: {
    frame: { id: 'main', url: `${RUN}/form` },
    childFrames: [
      { frame: { id: 'srcdoc', parentId: 'main', url: 'about:srcdoc' } },
      {
        frame: { id: 'other', parentId: 'main', url: `${OTHER}/frame` },
        childFrames: [
          { frame: { id: 'deep', parentId: 'other', url: 'about:blank' } },
        ],
      },
    ],
  },
};

describe('framesOfTree and pruneFrameTree', () => {
  it('lists every frame with its parent', () => {
    expect(framesOfTree(tree)).toEqual([
      ['main', { url: `${RUN}/form`, parentId: undefined }],
      ['srcdoc', { url: 'about:srcdoc', parentId: 'main' }],
      ['other', { url: `${OTHER}/frame`, parentId: 'main' }],
      ['deep', { url: 'about:blank', parentId: 'other' }],
    ]);
  });

  it('removes out-of-scope subtrees and never names their URLs', () => {
    const out = pruneFrameTree(tree, keepOnly('main', 'srcdoc'));
    expect(JSON.stringify(out)).not.toContain(OTHER);
    expect(JSON.stringify(out)).not.toContain('deep');
    expect(
      (out as typeof tree).frameTree.childFrames.map((c) => c.frame.id),
    ).toEqual(['srcdoc']);
  });

  it('is null when the root itself is out of scope', () => {
    expect(pruneFrameTree(tree, () => false)).toBeNull();
  });
});

describe('pruneDescribedNode', () => {
  it('strips an out-of-scope frame owner of its frame and document', () => {
    const described = {
      node: {
        nodeName: 'BODY',
        children: [
          {
            nodeName: 'IFRAME',
            frameId: 'other',
            contentDocument: {
              documentURL: `${OTHER}/frame`,
              children: [{ attributes: ['value', 'secret'] }],
            },
          },
          {
            nodeName: 'IFRAME',
            frameId: 'srcdoc',
            contentDocument: { documentURL: 'about:srcdoc' },
          },
        ],
      },
    };
    const out = pruneDescribedNode(described, keepOnly('srcdoc'));
    const text = JSON.stringify(out);
    expect(text).not.toContain(OTHER);
    expect(text).not.toContain('secret');
    expect(text).not.toContain('"other"');
    expect(text).toContain('about:srcdoc');
    // The input is not modified.
    expect(JSON.stringify(described)).toContain('secret');
  });

  it('walks shadow roots, templates and pseudo elements too', () => {
    const out = pruneDescribedNode(
      {
        node: {
          shadowRoots: [
            { children: [{ frameId: 'x', contentDocument: { a: 'hidden' } }] },
          ],
          templateContent: { frameId: 'y', contentDocument: { a: 'hidden' } },
        },
      },
      () => false,
    );
    expect(JSON.stringify(out)).not.toContain('hidden');
  });
});

describe('scopeFrameEvent', () => {
  const fresh = () => {
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    return scope;
  };
  const navigated = (id: string, url: string, parentId?: string) => ({
    frame: { id, url, ...(parentId ? { parentId } : {}) },
  });

  it('tracks a subframe from attach to an out-of-scope commit, then drops its events', () => {
    const scope = fresh();
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameAttached', {
        frameId: 'f',
        parentFrameId: 'main',
      }),
    ).toBe(true);
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Page.frameNavigated',
        navigated('f', `${OTHER}/x`, 'main'),
      ),
    ).toBe(false);
    expect(scope.frames.get('f')).toEqual({
      url: `${OTHER}/x`,
      parentId: 'main',
    });
    for (const method of [
      'Page.frameStartedLoading',
      'Page.frameStoppedLoading',
      'Page.lifecycleEvent',
    ]) {
      expect(scopeFrameEvent(scope, origins, method, { frameId: 'f' })).toBe(
        false,
      );
    }
    // The agent saw it attach, so it hears of the detach (ids only).
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameDetached', { frameId: 'f' }),
    ).toBe(true);
    expect(scope.frames.has('f')).toBe(false);
  });

  it('adds a frame first seen in frameNavigated (swapped in from another process)', () => {
    const scope = fresh();
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Page.frameNavigated',
        navigated('g', `${RUN}/ok`, 'main'),
      ),
    ).toBe(true);
    expect(scope.frames.get('g')?.url).toBe(`${RUN}/ok`);
  });

  it('judges pre-commit navigation events by their destination', () => {
    const scope = fresh();
    scope.frames.set('f', { url: '', parentId: 'main' });
    for (const method of [
      'Page.frameRequestedNavigation',
      'Page.frameStartedNavigating',
      'Page.frameScheduledNavigation',
    ]) {
      expect(
        scopeFrameEvent(scope, origins, method, {
          frameId: 'f',
          url: `${OTHER}/x`,
        }),
      ).toBe(false);
      expect(
        scopeFrameEvent(scope, origins, method, {
          frameId: 'f',
          url: `${RUN}/y`,
        }),
      ).toBe(true);
      expect(
        scopeFrameEvent(scope, origins, method, {
          frameId: 'f',
          url: 'javascript:void(0)',
        }),
      ).toBe(true);
    }
    const request = (url: string, type = 'Document') => ({
      frameId: 'f',
      type,
      request: { url },
    });
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Network.requestWillBeSent',
        request(`${OTHER}/x`),
      ),
    ).toBe(false);
    expect(
      scopeFrameEvent(scope, origins, 'Network.responseReceived', {
        frameId: 'f',
        type: 'Document',
        response: { url: `${OTHER}/x` },
      }),
    ).toBe(false);
    // A subresource fetched by an in-scope frame is judged by its frame only.
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Network.requestWillBeSent',
        request(`${OTHER}/api`, 'Fetch'),
      ),
    ).toBe(true);
  });

  it('updates the URL on navigatedWithinDocument', () => {
    const scope = fresh();
    scope.frames.set('f', { url: `${RUN}/a`, parentId: 'main' });
    expect(
      scopeFrameEvent(scope, origins, 'Page.navigatedWithinDocument', {
        frameId: 'f',
        url: `${RUN}/b`,
      }),
    ).toBe(true);
    expect(scope.frames.get('f')?.url).toBe(`${RUN}/b`);
  });

  it('maps execution contexts to frames and drops out-of-scope ones', () => {
    const scope = fresh();
    scope.frames.set('f', { url: `${OTHER}/x`, parentId: 'main' });
    const created = (id: number, frameId: string, origin: string) => ({
      context: {
        id,
        uniqueId: `u${id}`,
        origin,
        name: '',
        auxData: { frameId, isDefault: true },
      },
    });
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Runtime.executionContextCreated',
        created(3, 'main', RUN),
      ),
    ).toBe(true);
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Runtime.executionContextCreated',
        created(4, 'f', OTHER),
      ),
    ).toBe(false);
    expect(scope.contexts.get(3)).toBe('main');
    expect(scope.contexts.get(4)).toBe('f');
    expect(scope.uniqueContexts.get('u4')).toBe(4);
    // A context whose origin is foreign is dropped even when its frame looks
    // in scope (the context can be created before frameNavigated).
    scope.frames.set('late', { url: '', parentId: 'main' });
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Runtime.executionContextCreated',
        created(5, 'late', OTHER),
      ),
    ).toBe(false);
    // Isolated worlds report an empty origin.
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Runtime.executionContextCreated',
        created(6, 'main', ''),
      ),
    ).toBe(true);
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextDestroyed', {
        executionContextId: 4,
        executionContextUniqueId: 'u4',
      }),
    ).toBe(false);
    expect(scope.contexts.has(4)).toBe(false);
    expect(scope.uniqueContexts.has('u4')).toBe(false);
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextDestroyed', {
        executionContextId: 3,
      }),
    ).toBe(true);
    // Unknown: nothing to say about it.
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextDestroyed', {
        executionContextId: 99,
      }),
    ).toBe(false);
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Runtime.executionContextCreated',
        created(7, 'main', RUN),
      ),
    ).toBe(true);
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextsCleared', {}),
    ).toBe(true);
    expect(scope.contexts.size).toBe(0);
    expect(scope.uniqueContexts.size).toBe(0);
  });

  it('passes events that name no frame', () => {
    const scope = fresh();
    expect(
      scopeFrameEvent(scope, origins, 'Page.loadEventFired', {
        timestamp: 1,
      }),
    ).toBe(true);
    expect(
      scopeFrameEvent(scope, origins, 'Network.loadingFinished', {
        requestId: '1',
      }),
    ).toBe(true);
  });
});

describe('navigationDestination and destinationAllowed', () => {
  it('reads pre-commit and Document network destinations only', () => {
    expect(
      navigationDestination('Page.frameRequestedNavigation', { url: 'x' }),
    ).toBe('x');
    expect(
      navigationDestination('Network.requestWillBeSent', {
        type: 'Document',
        request: { url: 'y' },
      }),
    ).toBe('y');
    expect(
      navigationDestination('Network.requestWillBeSent', {
        type: 'XHR',
        request: { url: 'y' },
      }),
    ).toBeUndefined();
    expect(
      navigationDestination('Page.frameStartedNavigating', { url: 3 }),
    ).toBe('');
    expect(navigationDestination('Page.frameNavigated', {})).toBeUndefined();
  });

  it('allows in-scope, inheriting and javascript: destinations only', () => {
    expect(destinationAllowed(`${RUN}/x`, origins)).toBe(true);
    expect(destinationAllowed('about:blank', origins)).toBe(true);
    expect(destinationAllowed('javascript:void(0)', origins)).toBe(true);
    expect(destinationAllowed(`${OTHER}/x`, origins)).toBe(false);
    expect(destinationAllowed('', origins)).toBe(false);
    expect(
      destinationAllowed(`${OTHER}/x`, {
        origin: RUN,
        allowedFrameOrigins: [OTHER],
      }),
    ).toBe(true);
  });
});

describe('scopeFrameEvent alsoKeep', () => {
  it('admits a frame judged elsewhere, and only that one', () => {
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    const event = (frameId: string) =>
      scopeFrameEvent(
        scope,
        origins,
        'Page.frameStartedLoading',
        { frameId },
        (id) => id === 'oopif',
      );
    expect(event('oopif')).toBe(true);
    expect(event('other')).toBe(false);
  });
});

describe('scopeFrameEvent end events follow what the agent was told', () => {
  const setup = () => {
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    return scope;
  };
  const created = (id: number, frameId: string) => ({
    context: { id, origin: RUN, name: '', auxData: { frameId } },
  });

  it('forwards the destroy of an announced context after its frame detached', () => {
    const scope = setup();
    scopeFrameEvent(scope, origins, 'Page.frameAttached', {
      frameId: 'f',
      parentFrameId: 'main',
    });
    expect(
      scopeFrameEvent(
        scope,
        origins,
        'Runtime.executionContextCreated',
        created(8, 'f'),
      ),
    ).toBe(true);
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameDetached', { frameId: 'f' }),
    ).toBe(true);
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextDestroyed', {
        executionContextId: 8,
      }),
    ).toBe(true);
  });

  it('forwards the destroy of an announced context after its frame left scope', () => {
    const scope = setup();
    scope.frames.set('f', { url: `${RUN}/a`, parentId: 'main' });
    scopeFrameEvent(
      scope,
      origins,
      'Runtime.executionContextCreated',
      created(9, 'f'),
    );
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameNavigated', {
        frame: { id: 'f', parentId: 'main', url: `${OTHER}/x` },
      }),
    ).toBe(false);
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextDestroyed', {
        executionContextId: 9,
      }),
    ).toBe(true);
    // Never told of, never ended: a context of the out-of-scope document.
    scopeFrameEvent(
      scope,
      origins,
      'Runtime.executionContextCreated',
      created(10, 'f'),
    );
    expect(
      scopeFrameEvent(scope, origins, 'Runtime.executionContextDestroyed', {
        executionContextId: 10,
      }),
    ).toBe(false);
  });

  it('forwards the detach of a frame the agent saw attach, wherever it went', () => {
    const scope = setup();
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameAttached', {
        frameId: 'g',
        parentFrameId: 'main',
      }),
    ).toBe(true);
    scopeFrameEvent(scope, origins, 'Page.frameNavigated', {
      frame: { id: 'g', parentId: 'main', url: `${OTHER}/x` },
    });
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameDetached', { frameId: 'g' }),
    ).toBe(true);
    // A frame the agent never heard of stays unheard of.
    scope.frames.set('h', { url: `${OTHER}/y`, parentId: 'main' });
    expect(
      scopeFrameEvent(scope, origins, 'Page.frameDetached', { frameId: 'h' }),
    ).toBe(false);
  });
});

describe('commandAddresses with several handles', () => {
  it('collects every node handle, not only the first', () => {
    expect(commandAddresses({ backendNodeId: 1, nodeId: 2 }).nodes).toEqual([
      { backendNodeId: 1 },
      { nodeId: 2 },
    ]);
  });

  it('collects every handle kind together', () => {
    const a = commandAddresses({
      objectId: '1.3.1',
      executionContextId: 4,
      backendNodeId: 5,
      frameId: 'f',
    });
    expect(a.objectIds).toEqual(['1.3.1']);
    expect(a.contextIds).toEqual([4]);
    expect(a.nodes).toEqual([{ backendNodeId: 5 }]);
    expect(a.frameIds).toEqual(['f']);
  });
});

describe('contextInScope', () => {
  const setup = () => {
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    scope.frames.set('f', { url: `${RUN}/inner`, parentId: 'main' });
    return scope;
  };
  const created = (id: number, frameId: string, origin: string) => ({
    context: { id, origin, name: '', auxData: { frameId } },
  });

  it('is false for an unknown context and true for an in-scope one', () => {
    const scope = setup();
    scopeFrameEvent(
      scope,
      origins,
      'Runtime.executionContextCreated',
      created(3, 'f', RUN),
    );
    expect(contextInScope(scope, origins, 3)).toBe(true);
    expect(contextInScope(scope, origins, 99)).toBe(false);
  });

  it('refuses a context created in scope after its frame navigates cross-origin', () => {
    const scope = setup();
    scopeFrameEvent(
      scope,
      origins,
      'Runtime.executionContextCreated',
      created(3, 'f', RUN),
    );
    scopeFrameEvent(scope, origins, 'Page.frameNavigated', {
      frame: { id: 'f', parentId: 'main', url: `${OTHER}/x` },
    });
    expect(contextInScope(scope, origins, 3)).toBe(false);
    // Even if the frame comes back in scope later, the old context stays
    // refused: it belonged to a document of another origin's era.
    scopeFrameEvent(scope, origins, 'Page.frameNavigated', {
      frame: { id: 'f', parentId: 'main', url: `${RUN}/back` },
    });
    expect(contextInScope(scope, origins, 3)).toBe(false);
  });

  it('refuses a context whose own origin is out of scope while its frame still looks in scope', () => {
    const scope = setup();
    scopeFrameEvent(
      scope,
      origins,
      'Runtime.executionContextCreated',
      created(4, 'f', OTHER),
    );
    expect(frameInScope(scope.frames, 'f', RUN, [])).toBe(true);
    expect(contextInScope(scope, origins, 4)).toBe(false);
  });

  it('keeps an isolated world (no origin) on its frame verdict', () => {
    const scope = setup();
    scopeFrameEvent(
      scope,
      origins,
      'Runtime.executionContextCreated',
      created(5, 'f', ''),
    );
    expect(contextInScope(scope, origins, 5)).toBe(true);
  });

  it('honours an origin recorded by discovery', () => {
    const scope = setup();
    scope.contexts.set(6, 'f');
    scope.contextOrigins.set(6, documentOrigin(`${OTHER}/x`) as string);
    expect(contextInScope(scope, origins, 6)).toBe(false);
    scope.contexts.set(7, 'f');
    expect(contextInScope(scope, origins, 7)).toBe(true);
  });
});

describe('documentOrigin', () => {
  it('is the https origin, undefined when inherited, a refusal otherwise', () => {
    expect(documentOrigin(`${RUN}/x?y`)).toBe(RUN);
    expect(documentOrigin('about:srcdoc')).toBeUndefined();
    expect(documentOrigin('about:blank')).toBeUndefined();
    const refused = documentOrigin('data:text/html,x');
    expect(refused).toBeDefined();
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    scope.contexts.set(1, 'main');
    scope.contextOrigins.set(1, refused as string);
    expect(contextInScope(scope, origins, 1)).toBe(false);
  });
});

describe('dialog events are not frame-scoped', () => {
  it('forwards a dialog raised by an out-of-scope frame (filterEvent blanks it)', () => {
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    scope.frames.set('f', { url: `${OTHER}/x`, parentId: 'main' });
    for (const method of [
      'Page.javascriptDialogOpening',
      'Page.javascriptDialogClosed',
    ]) {
      expect(
        scopeFrameEvent(scope, origins, method, {
          frameId: 'f',
          type: 'alert',
        }),
      ).toBe(true);
    }
  });
});

describe('network follow-ups of a dropped request', () => {
  it('are dropped too, and forgotten once the request ends', () => {
    const scope = createFrameScope();
    scope.frames.set('main', { url: `${RUN}/form` });
    scope.frames.set('f', { url: `${OTHER}/x`, parentId: 'main' });
    expect(
      scopeFrameEvent(scope, origins, 'Network.requestWillBeSent', {
        requestId: 'r1',
        frameId: 'f',
        type: 'Script',
        request: { url: `${OTHER}/a.js` },
      }),
    ).toBe(false);
    for (const method of [
      'Network.requestServedFromCache',
      'Network.loadingFinished',
    ]) {
      expect(scopeFrameEvent(scope, origins, method, { requestId: 'r1' })).toBe(
        false,
      );
    }
    expect(scope.droppedRequests.has('r1')).toBe(false);
    expect(
      scopeFrameEvent(scope, origins, 'Network.loadingFailed', {
        requestId: 'r2',
      }),
    ).toBe(true);
  });

  it('keeps the dropped set bounded', () => {
    const scope = createFrameScope();
    scope.frames.set('f', { url: `${OTHER}/x` });
    for (let i = 0; i < 5000; i++) {
      scopeFrameEvent(scope, origins, 'Network.requestWillBeSent', {
        requestId: `r${i}`,
        frameId: 'f',
        type: 'Script',
        request: { url: `${OTHER}/a.js` },
      });
    }
    expect(scope.droppedRequests.size).toBeLessThanOrEqual(2000);
    expect(scope.droppedRequests.has('r4999')).toBe(true);
  });
});

describe('result classification', () => {
  it('classifies every allowed CDP method exactly once', () => {
    const pruned = [...PRUNED_RESULT_METHODS];
    const plain = [...NO_FRAME_CONTENT_METHODS];
    expect(pruned.filter((m) => NO_FRAME_CONTENT_METHODS.has(m))).toEqual([]);
    expect([...pruned, ...plain].sort()).toEqual(
      [...ALLOWED_CDP_METHODS].sort(),
    );
  });

  // Pinned per allowed method: the params that address a frame, context,
  // object or node. A new allowed method fails here until it is classified.
  const HANDLE_PARAMS: Record<string, Record<string, unknown>> = {
    'Runtime.evaluate': { contextId: 1, uniqueContextId: 'u' },
    'Runtime.enable': {},
    'Runtime.callFunctionOn': {
      objectId: '1.2.3',
      executionContextId: 1,
      arguments: [{ objectId: '1.2.4' }],
    },
    'Runtime.runIfWaitingForDebugger': {},
    'Page.enable': {},
    'Page.navigate': { frameId: 'f' },
    'Page.captureScreenshot': {},
    'Page.handleJavaScriptDialog': {},
    'Page.getFrameTree': {},
    'Page.bringToFront': {},
    'Network.enable': {},
    'Target.setAutoAttach': {},
    'DOM.enable': {},
    'DOM.describeNode': { nodeId: 1, backendNodeId: 2, objectId: '1.2.3' },
    'DOM.resolveNode': {
      nodeId: 1,
      backendNodeId: 2,
      executionContextId: 3,
    },
    'DOM.getBoxModel': { nodeId: 1, backendNodeId: 2, objectId: '1.2.3' },
    'DOM.scrollIntoViewIfNeeded': {
      nodeId: 1,
      backendNodeId: 2,
      objectId: '1.2.3',
    },
    'Accessibility.enable': {},
    'Accessibility.getFullAXTree': { frameId: 'f' },
    'Input.insertText': {},
    'Input.dispatchKeyEvent': {},
    'Input.dispatchMouseEvent': {},
  };

  it('pins the handle params of every allowed method', () => {
    expect(Object.keys(HANDLE_PARAMS).sort()).toEqual(
      [...ALLOWED_CDP_METHODS].sort(),
    );
    for (const params of Object.values(HANDLE_PARAMS)) {
      const a = commandAddresses(params);
      const found =
        a.frameIds.length +
        a.contextIds.length +
        a.uniqueContextIds.length +
        a.objectIds.length +
        a.nodes.length;
      const expected = Object.entries(params).reduce(
        (n, [key, value]) => n + (Array.isArray(value) ? value.length : 1),
        0,
      );
      expect(found).toBe(expected);
    }
  });
});
