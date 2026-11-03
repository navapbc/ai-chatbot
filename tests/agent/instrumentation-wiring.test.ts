import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const registerOTel = vi.fn();

vi.mock('@vercel/otel', () => ({ registerOTel }));
vi.mock('eve/instrumentation', () => ({
  defineInstrumentation: (config: unknown) => config,
}));
vi.mock('@braintrust/otel', () => ({
  BraintrustExporter: class {
    export() {}
    async shutdown() {}
    async forceFlush() {}
  },
}));

type Setup = { setup: (ctx: { agentName: string }) => void };

async function runSetup() {
  const mod = (await import('@/agent/instrumentation')) as unknown as {
    default: Setup;
  };
  mod.default.setup({ agentName: 'agent' });
}

const ENV_KEYS = [
  'BRAINTRUST_API_KEY',
  'RELAY_RUNS_ENABLED',
  'OTEL_EXPORTER_OTLP_ENDPOINT',
  'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT',
] as const;

describe('agent/instrumentation', () => {
  const saved: Record<string, string | undefined> = {};
  beforeEach(() => {
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    process.env.BRAINTRUST_API_KEY = 'test-key';
    registerOTel.mockClear();
  });
  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
    vi.restoreAllMocks();
  });

  it('registers only explicit span processors wrapping the scrubbing exporter, with no traceExporter', async () => {
    await runSetup();
    expect(registerOTel).toHaveBeenCalledOnce();
    const options = registerOTel.mock.calls[0][0] as Record<string, unknown>;
    expect(options).not.toHaveProperty('traceExporter');
    const processors = options.spanProcessors as Array<{
      _exporter?: { constructor: { name: string } };
    }>;
    expect(Array.isArray(processors)).toBe(true);
    expect(processors).toHaveLength(1);
    expect(processors[0]._exporter?.constructor.name).toBe(
      'ScrubbingSpanExporter',
    );
  });

  it('warns once when the flag is on and an OTLP endpoint variable is set', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    process.env.RELAY_RUNS_ENABLED = '1';
    process.env.OTEL_EXPORTER_OTLP_ENDPOINT = 'http://collector.invalid';
    await runSetup();
    expect(warn).toHaveBeenCalledOnce();
    expect(String(warn.mock.calls[0][0])).not.toContain('collector.invalid');
  });

  it('does not warn when the flag is off or no endpoint variable is set', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = 'http://collector.invalid';
    await runSetup();
    Reflect.deleteProperty(process.env, 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT');
    process.env.RELAY_RUNS_ENABLED = '1';
    await runSetup();
    expect(warn).not.toHaveBeenCalled();
  });
});
