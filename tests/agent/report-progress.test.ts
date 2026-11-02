import type { ToolContext } from 'eve/tools';
import { afterEach, describe, expect, it, vi } from 'vitest';

type Principal = {
  attributes: Record<string, string>;
  authenticator: string;
  principalId: string;
  principalType: 'user';
  subject: string;
};
const principal = (attributes: Record<string, string>): Principal => ({
  attributes,
  authenticator: 'test',
  principalId: 'p',
  principalType: 'user',
  subject: 's',
});
const ctxWith = (attributes: Record<string, string>) =>
  ({
    session: {
      id: 's1',
      auth: { initiator: principal(attributes), current: null },
    },
  }) as unknown as ToolContext;

type Tool = {
  inputSchema: {
    safeParse: (v: unknown) => { success: boolean; data?: unknown };
  };
  execute: (i: unknown, c: ToolContext) => Promise<unknown>;
};

const RUN = '0b6f7c4e-5d1a-4c55-9a53-3f7d2f0c9a11';
const REFUSAL =
  'Not available in extension runs: use ask_question for missing data and report_progress for status.';

describe('report_progress tool', () => {
  it('parses a valid report, strips extra keys, and echoes only ok', async () => {
    const { default: tool } = await import('@/agent/tools/report_progress');
    const t = tool as unknown as Tool;
    const parsed = t.inputSchema.safeParse({
      sections: [
        { name: 'Name', status: 'done', filled: 2, value: '123-45-6789' },
      ],
      current: 'Address',
      extra: 'secret',
    });
    expect(parsed.success).toBe(true);
    expect(JSON.stringify(parsed.data)).not.toContain('123-45');
    expect(JSON.stringify(parsed.data)).not.toContain('secret');
    expect(await t.execute(parsed.data, ctxWith({}))).toEqual({ ok: true });
  });

  it('rejects over-limit input', async () => {
    const { default: tool } = await import('@/agent/tools/report_progress');
    const t = tool as unknown as Tool;
    const sections = Array.from({ length: 41 }, (_, i) => ({
      name: `s${i}`,
      status: 'waiting',
      filled: 0,
    }));
    expect(t.inputSchema.safeParse({ sections }).success).toBe(false);
    expect(
      t.inputSchema.safeParse({
        sections: [{ name: 'x'.repeat(81), status: 'done', filled: 1 }],
      }).success,
    ).toBe(false);
  });
});

describe.each([
  [
    'form_review form_summary',
    'subagents/form_review/tools/form_summary',
    { fields: [{ field: 'a', source: 'missing' }] },
  ],
])('%s on relay sessions', (_name, path, input) => {
  afterEach(() => {
    vi.doUnmock('@/lib/jev/enrich');
    vi.doUnmock('@/agent/lib/jev');
    vi.resetModules();
  });

  const load = async () => {
    vi.resetModules();
    const annotate = vi.fn(async ({ fields }: { fields: unknown[] }) => fields);
    const participant = vi.fn(() => null);
    vi.doMock('@/lib/jev/enrich', () => ({
      annotateGapFields: annotate,
      annotateSummaryFields: annotate,
    }));
    vi.doMock('@/agent/lib/jev', () => ({
      currentParticipant: participant,
      currentCaseworkerMessages: () => [],
    }));
    const { default: tool } = await import(`@/agent/${path}`);
    return { tool: tool as unknown as Tool, annotate, participant };
  };

  it.each([
    ['complete', { relayUrl: 'wss://r.example.gov', relayRunId: RUN }],
    ['partial', { relayRunId: RUN }],
  ])('refuses with %s credentials before any Jev work', async (_k, attrs) => {
    const { tool, annotate, participant } = await load();
    expect(await tool.execute(input, ctxWith(attrs))).toEqual({
      success: false,
      error: REFUSAL,
    });
    expect(annotate).not.toHaveBeenCalled();
    expect(participant).not.toHaveBeenCalled();
  });

  it('runs normally with no relay credentials', async () => {
    const { tool, annotate } = await load();
    const result = (await tool.execute(input, ctxWith({}))) as {
      rendered?: boolean;
    };
    expect(result.rendered).toBe(true);
    expect(annotate).toHaveBeenCalledTimes(1);
  });
});

describe('gap_analysis and form_summary on relay sessions', () => {
  afterEach(() => {
    vi.doUnmock('@/lib/jev/enrich');
    vi.doUnmock('@/agent/lib/jev');
    vi.resetModules();
  });

  const relay = { relayUrl: 'wss://r.example.gov', relayRunId: RUN };

  const load = async (path: string) => {
    vi.resetModules();
    const annotate = vi.fn(async ({ fields }: { fields: unknown[] }) => fields);
    vi.doMock('@/lib/jev/enrich', () => ({
      annotateGapFields: annotate,
      annotateSummaryFields: annotate,
    }));
    vi.doMock('@/agent/lib/jev', () => ({
      currentParticipant: () => null,
      currentCaseworkerMessages: () => [
        'she lives in Riverside',
        'her number is (951) 555-0100',
        'ssn 123-45-6789',
        'ref 4111 1111 1111 1111',
        'id １２３４５６７８９',
        'id 123 / 45 // 6789',
        'id ١٢٣٤٥٦٧٨٩',
        'id 1 . . . 2 - - 3 _ _ 4 5 6 7 8 9',
        'id 12\u200B345\u200B6789',
      ],
    }));
    const { default: tool } = await import(`@/agent/${path}`);
    return { tool: tool as unknown as Tool, annotate };
  };

  it('form_summary sends only turns with no SSN, long number or phone number', async () => {
    const { tool, annotate } = await load('tools/form_summary');
    await tool.execute(
      { fields: [{ field: 'Phone', value: 'x', source: 'caseworker' }] },
      ctxWith(relay),
    );
    const sent = (annotate.mock.calls[0][0] as { caseworkerMessages: string[] })
      .caseworkerMessages;
    expect(sent).toEqual(['she lives in Riverside']);
  });

  it('gap_analysis runs, sends no note, and does not end the turn', async () => {
    const { tool, annotate } = await load('tools/gap_analysis');
    const result = (await tool.execute(
      {
        formName: 'F',
        missingFields: [{ field: 'Phone', note: 'caseworker said 555-0100' }],
      },
      ctxWith(relay),
    )) as { rendered?: boolean; next?: string };
    expect(result.rendered).toBe(true);
    expect(result.next).toContain('ask_question');
    const sent = annotate.mock.calls[0][0] as { fields: object[] };
    expect(JSON.stringify(sent.fields)).not.toContain('555-0100');
  });

  it('form_summary runs and removes values of sensitive fields', async () => {
    const { tool, annotate } = await load('tools/form_summary');
    const result = (await tool.execute(
      {
        fields: [
          { field: 'Phone', value: '(951) 555-0100', source: 'record' },
          {
            field: 'Social Security Number',
            value: '123-45-6789',
            source: 'record',
          },
          { field: 'Notes', value: 'ref 123-45-6789', source: 'inferred' },
        ],
      },
      ctxWith(relay),
    )) as { rendered?: boolean };
    expect(result.rendered).toBe(true);
    const sent = JSON.stringify(
      (annotate.mock.calls[0][0] as { fields: object[] }).fields,
    );
    expect(sent).toContain('(951) 555-0100');
    expect(sent).not.toContain('123-45-6789');
  });
});
