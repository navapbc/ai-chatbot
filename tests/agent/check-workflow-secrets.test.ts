import zlib from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { scanBuffer } from '@/scripts/check-workflow-secrets';

const NEEDLE = 'TOKEN-needle-1234';
const CONTROL = 'control-string-5678';
const payload = `{"a":"${NEEDLE}","b":"${CONTROL}"}`;
const needles = [NEEDLE, CONTROL, 'absent-value'];

describe('scanBuffer', () => {
  it('finds needles in a plain buffer', () => {
    const found = scanBuffer(Buffer.from(payload), needles);
    expect(found.get(NEEDLE)).toBe(1);
    expect(found.get(CONTROL)).toBe(1);
  });

  it('finds needles inside a zstd-compressed buffer', () => {
    const found = scanBuffer(
      zlib.zstdCompressSync(Buffer.from(payload)),
      needles,
    );
    expect(found.get(NEEDLE)).toBe(1);
    expect(found.get(CONTROL)).toBe(1);
  });

  it('finds needles in a compressed frame that follows a header', () => {
    const buf = Buffer.concat([
      Buffer.from('zstd('),
      zlib.zstdCompressSync(Buffer.from(payload)),
    ]);
    const found = scanBuffer(buf, needles);
    expect(found.get(NEEDLE)).toBe(1);
    expect(found.get(CONTROL)).toBe(1);
  });

  it('finds needles in a frame nested inside another frame', () => {
    const inner = zlib.zstdCompressSync(Buffer.from(payload));
    const outer = zlib.zstdCompressSync(
      Buffer.concat([Buffer.from('h'), inner]),
    );
    expect(scanBuffer(outer, needles).get(NEEDLE)).toBe(1);
  });

  it('reports zero for an absent needle and ignores a corrupt frame', () => {
    const corrupt = Buffer.concat([
      Buffer.from([0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x01]),
      Buffer.from('x'),
    ]);
    const found = scanBuffer(corrupt, needles);
    expect(found.get('absent-value')).toBe(0);
    expect(found.get(NEEDLE)).toBe(0);
    expect(scanBuffer(Buffer.from(payload), needles).get('absent-value')).toBe(
      0,
    );
  });
});
