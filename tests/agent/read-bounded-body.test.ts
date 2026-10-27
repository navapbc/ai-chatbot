import { describe, expect, it } from 'vitest';
import { readBoundedText } from '@/lib/relay/read-bounded-body';

const post = (body: BodyInit | null, headers?: Record<string, string>) =>
  new Request('http://localhost/x', {
    method: 'POST',
    body,
    headers,
    // @ts-expect-error node fetch needs duplex for stream bodies
    duplex: 'half',
  });

describe('readBoundedText', () => {
  it('reads a body under the cap', async () => {
    expect(await readBoundedText(post('hello'), 10)).toEqual({
      ok: true,
      text: 'hello',
    });
  });
  it('accepts exactly the cap and refuses one byte over', async () => {
    expect((await readBoundedText(post('a'.repeat(10)), 10)).ok).toBe(true);
    expect((await readBoundedText(post('a'.repeat(11)), 10)).ok).toBe(false);
  });
  it('counts bytes, not characters', async () => {
    const body = 'é'.repeat(1100);
    expect(body).toHaveLength(1100);
    expect((await readBoundedText(post(body), 2048)).ok).toBe(false);
    expect((await readBoundedText(post(body), 2200)).ok).toBe(true);
  });
  it('returns empty text for no body', async () => {
    expect(await readBoundedText(post(null), 10)).toEqual({
      ok: true,
      text: '',
    });
  });

  const chunked = (chunks: number, size: number) => {
    let pulled = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pulled >= chunks) return controller.close();
        pulled++;
        controller.enqueue(new Uint8Array(size).fill(97));
      },
    });
    return { stream, pulls: () => pulled };
  };

  it('rejects a chunked body without content-length and stops reading', async () => {
    const { stream, pulls } = chunked(1000, 10);
    const res = await readBoundedText(post(stream), 100);
    expect(res.ok).toBe(false);
    expect(pulls()).toBeLessThan(100);
  });
  it('rejects a large stream behind a lying small content-length', async () => {
    const { stream, pulls } = chunked(1000, 10);
    const res = await readBoundedText(
      post(stream, { 'content-length': '5' }),
      100,
    );
    expect(res.ok).toBe(false);
    expect(pulls()).toBeLessThan(100);
  });
  it('refuses an over-cap content-length before reading', async () => {
    const { stream, pulls } = chunked(5, 10);
    const res = await readBoundedText(
      post(stream, { 'content-length': '999999' }),
      100,
    );
    expect(res.ok).toBe(false);
    expect(pulls()).toBe(0);
  });
});
