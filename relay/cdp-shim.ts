import type { RelayToExtension } from '../lib/relay/protocol';

// Presents a browser-level CDP endpoint on top of the extension, which can
// only see tabs. Ported from spike/cdp-relay/relay.mjs.

export type DistributiveOmit<T, K extends PropertyKey> = T extends unknown
  ? Omit<T, K>
  : never;

/** Sends to the run's extension and resolves with its reply result. */
export type ExtensionCall = (
  msg: DistributiveOmit<RelayToExtension, 'id'>,
) => Promise<unknown>;

export interface ShimHooks {
  /** Sends a browser-level event to the agent, after the command's reply. */
  emit?: (event: { method: string; params: unknown }) => void;
  /** Called with the name of a browser-level method the shim stubs out. */
  onUnknown?: (method: string) => void;
}

type TargetInfo = Record<string, unknown> & { targetId: string };

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Keeps only well-formed target entries from an extension `listTargets`
 * reply. Never adds an entry the extension did not return.
 */
export function sanitizeTargetInfos(result: unknown): TargetInfo[] {
  if (!isObject(result) || !Array.isArray(result.targetInfos)) return [];
  return result.targetInfos.filter(
    (t): t is TargetInfo => isObject(t) && typeof t.targetId === 'string',
  );
}

function requireTargetId(params: unknown): string {
  const targetId = isObject(params) ? params.targetId : undefined;
  if (typeof targetId !== 'string')
    throw new Error('targetId must be a string');
  return targetId;
}

const VERSION = {
  protocolVersion: '1.3',
  product: 'Chrome/relay',
  revision: '0',
  userAgent: 'relay',
  jsVersion: '0',
};

export async function handleBrowserLevel(
  method: string,
  params: unknown,
  ext: ExtensionCall,
  hooks: ShimHooks = {},
): Promise<unknown> {
  switch (method) {
    case 'Browser.getVersion':
      return { ...VERSION };
    case 'Target.getTargets':
      return {
        targetInfos: sanitizeTargetInfos(await ext({ type: 'listTargets' })),
      };
    case 'Target.attachToTarget': {
      const targetId = requireTargetId(params);
      const reply = await ext({ type: 'attach', targetId });
      if (!isObject(reply) || typeof reply.sessionId !== 'string') {
        throw new Error('extension returned no session');
      }
      const sessionId = reply.sessionId;
      const targetInfo = isObject(reply.targetInfo)
        ? reply.targetInfo
        : { targetId };
      // The spike sends this after the reply; agent-browser accepts it.
      const emit = hooks.emit;
      if (emit) {
        setTimeout(
          () =>
            emit({
              method: 'Target.attachedToTarget',
              params: {
                sessionId,
                targetInfo: { ...targetInfo, attached: true },
                waitingForDebugger: false,
              },
            }),
          0,
        );
      }
      return { sessionId };
    }
    case 'Target.createTarget': {
      const url = isObject(params) ? params.url : undefined;
      const reply = await ext({
        type: 'createTarget',
        url: typeof url === 'string' ? url : undefined,
      });
      if (!isObject(reply) || typeof reply.targetId !== 'string') {
        throw new Error('extension returned no target');
      }
      return { targetId: reply.targetId };
    }
    case 'Target.closeTarget': {
      const reply = await ext({
        type: 'closeTarget',
        targetId: requireTargetId(params),
      });
      return { success: isObject(reply) && reply.success === true };
    }
    case 'Target.getTargetInfo': {
      const targetId = isObject(params) ? params.targetId : undefined;
      const targets = sanitizeTargetInfos(await ext({ type: 'listTargets' }));
      const targetInfo =
        targetId === undefined
          ? targets[0]
          : targets.find((t) => t.targetId === targetId);
      if (!targetInfo) throw new Error('no such target');
      return { targetInfo };
    }
    default:
      hooks.onUnknown?.(method);
      return {};
  }
}
