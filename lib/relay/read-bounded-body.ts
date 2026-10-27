// Reads a request body as UTF-8 text without ever buffering more than
// `maxBytes`: a declared Content-Length over the cap is refused up front, and
// the stream is cancelled as soon as the running count passes it (a missing
// or understated Content-Length does not get past that).
export const readBoundedText = async (
  request: Request,
  maxBytes: number,
): Promise<{ ok: true; text: string } | { ok: false }> => {
  const declared = Number(request.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > maxBytes) return { ok: false };

  if (!request.body) return { ok: true, text: '' };

  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let total = 0;
  let text = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel().catch(() => {});
        return { ok: false };
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } catch {
    return { ok: false };
  }
  return { ok: true, text };
};
