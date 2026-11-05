import { describe, expect, it, vi } from 'vitest';
import {
  RESUME_MESSAGE,
  resumeAgentTurn,
} from '../../extension/panel/lib/resume-client';
import {
  AnswerInFlightError,
  RunEndedError,
} from '../../extension/panel/lib/copy';

const RUN = '11111111-2222-4333-8444-555555555555';
const ORIGIN = 'https://app.example.org';

const streamOf = (text: string) =>
  new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(text));
        controller.close();
      },
    }),
    { status: 200 },
  );

describe('resumeAgentTurn', () => {
  it('posts one fixed user message for the run and nothing else', async () => {
    const fetchMock = vi.fn(async () => streamOf('data: x\n\n'));
    const result = await resumeAgentTurn(
      { fetch: fetchMock as unknown as typeof fetch, appOrigin: ORIGIN },
      { runId: RUN },
      () => 'msg-1',
    );
    expect(result).toEqual({ ok: true });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe(`${ORIGIN}/api/eve-chat`);
    expect(JSON.parse(init.body as string)).toEqual({
      id: RUN,
      message: {
        id: 'msg-1',
        role: 'user',
        parts: [{ type: 'text', text: RESUME_MESSAGE }],
      },
    });
  });

  it('tells the agent to read the page again and not to overwrite', () => {
    expect(RESUME_MESSAGE).toMatch(/fresh snapshot/);
    expect(RESUME_MESSAGE).toMatch(/do not overwrite/);
    expect(RESUME_MESSAGE).toMatch(/report_progress/);
  });

  it('maps a run_ended refusal to the fixed ended error', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ code: 'bad_request:api', cause: 'run_ended' }),
          {
            status: 400,
          },
        ),
    );
    const result = await resumeAgentTurn(
      { fetch: fetchMock as unknown as typeof fetch, appOrigin: ORIGIN },
      { runId: RUN },
      () => 'm',
    );
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toBeInstanceOf(RunEndedError);
  });

  it('refuses a second resume while one is on its way', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => {
      release = r;
    });
    const fetchMock = vi.fn(async () => {
      await gate;
      return streamOf('data: x\n\n');
    });
    const deps = {
      fetch: fetchMock as unknown as typeof fetch,
      appOrigin: ORIGIN,
    };
    const first = resumeAgentTurn(deps, { runId: RUN }, () => 'a');
    const second = await resumeAgentTurn(deps, { runId: RUN }, () => 'b');
    expect(second.ok).toBe(false);
    if (!second.ok) expect(second.error).toBeInstanceOf(AnswerInFlightError);
    release();
    expect((await first).ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
