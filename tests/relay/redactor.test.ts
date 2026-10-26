import { describe, expect, it } from 'vitest';
import {
  createRedactor,
  descriptorFromAttributes,
  valueVariants,
} from '@/lib/relay/redactor';
import { type FieldDescriptor, MASK } from '@/lib/relay/sensitive-fields';

// Fixture AX nodes shaped like Chromium's Accessibility.getFullAXTree output
// (measured with Chromium via Playwright; see the Task 3 report).
type AXNode = Record<string, unknown>;

const str = (value: string) => ({ type: 'computedString', value });

const textbox = (
  nodeId: string,
  label: string,
  value: string,
  extra: Partial<AXNode> = {},
): AXNode => ({
  nodeId,
  ignored: false,
  role: { type: 'role', value: 'textbox' },
  name: str(label),
  value: { type: 'string', value },
  properties: [
    { name: 'focusable', value: { type: 'booleanOrUndefined', value: true } },
    { name: 'editable', value: { type: 'token', value: 'plaintext' } },
  ],
  childIds: [],
  ...extra,
});

const staticText = (nodeId: string, text: string, parentId: string) => ({
  nodeId,
  parentId,
  ignored: false,
  role: { type: 'role', value: 'StaticText' },
  name: str(text),
  childIds: [],
});

const tree = (...nodes: AXNode[]) => ({ nodes });

const nodesOf = (result: unknown) => (result as { nodes: AXNode[] }).nodes;
const byId = (result: unknown, id: string) =>
  nodesOf(result).find((n) => n.nodeId === id) as AXNode & {
    name?: { value: unknown };
    value?: { value: unknown };
    description?: { value: unknown };
    properties?: Array<{ name: string; value: { value: unknown } }>;
  };

const AX = 'Accessibility.getFullAXTree';

describe('valueVariants', () => {
  it('covers a 9-digit SSN in digits-only, hyphenated, spaced, dotted and slashed forms', () => {
    const v = valueVariants('123-45-6789');
    for (const form of [
      '123-45-6789',
      '123456789',
      '123 45 6789',
      '123.45.6789',
      '123/45/6789',
    ]) {
      expect(v).toContain(form);
    }
    expect(valueVariants('123456789')).toContain('123-45-6789');
    expect(valueVariants('123.45.6789')).toContain('123 45 6789');
  });

  it('covers a 16-digit card in 4-4-4-4 forms', () => {
    const v = valueVariants('4111111111111111');
    expect(v).toContain('4111-1111-1111-1111');
    expect(v).toContain('4111 1111 1111 1111');
    expect(valueVariants('4111 1111 1111 1111')).toContain('4111111111111111');
  });

  it('adds case variants and URL-encoded forms, without duplicates', () => {
    const v = valueVariants('Fake.User@Example.test');
    expect(v).toContain('fake.user@example.test');
    expect(v).toContain('FAKE.USER@EXAMPLE.TEST');
    expect(v).toContain('Fake.User%40Example.test');
    expect(new Set(v).size).toBe(v.length);
    // No digit extraction from a mixed value: that would scrub years and
    // page numbers everywhere.
    expect(valueVariants('user1234@x.test')).not.toContain('1234');
  });

  it('drops variants shorter than 4 characters', () => {
    expect(valueVariants('a-b-c').every((x) => x.length >= 4)).toBe(true);
  });
});

describe('createRedactor', () => {
  it('masks the value of a node whose descriptor is sensitive', async () => {
    const r = createRedactor();
    const out = await r.redactResult(
      AX,
      tree(textbox('1', 'Social Security Number', '123-45-6789')),
    );
    expect(byId(out, '1').value?.value).toBe(MASK);
  });

  it('masks name, description and valuetext of a sensitive node, not only value', async () => {
    const r = createRedactor();
    const node = textbox('1', 'SSN 123-45-6789', '123-45-6789', {
      description: str('ends 6789 for 123-45-6789'),
      properties: [
        { name: 'valuetext', value: { type: 'string', value: '123-45-6789' } },
        {
          name: 'focusable',
          value: { type: 'booleanOrUndefined', value: true },
        },
        { name: 'invalid', value: { type: 'token', value: 'false' } },
      ],
    });
    const out = byId(await r.redactResult(AX, tree(node)), '1');
    expect(out.name?.value).toBe(MASK);
    expect(out.description?.value).toBe(MASK);
    expect(out.value?.value).toBe(MASK);
    const props = Object.fromEntries(
      (out.properties ?? []).map((p) => [p.name, p.value.value]),
    );
    expect(props.valuetext).toBe(MASK);
    // Non-text properties keep their meaning.
    expect(props.focusable).toBe(true);
    expect(props.invalid).toBe('false');
  });

  it('masks a node whose sensitive value is only in valuetext', async () => {
    const r = createRedactor();
    const node = {
      nodeId: '1',
      ignored: false,
      role: { type: 'role', value: 'spinbutton' },
      name: str('PIN'),
      properties: [
        { name: 'valuetext', value: { type: 'string', value: '4321' } },
      ],
    };
    const out = byId(await r.redactResult(AX, tree(node)), '1');
    expect(out.properties?.[0].value.value).toBe(MASK);
  });

  it('masks the text descendants of a sensitive field (StaticText, InlineTextBox, option)', async () => {
    const r = createRedactor();
    const input = textbox('8', 'Social security ', '123-45-6789', {
      childIds: ['9'],
    });
    const generic = {
      nodeId: '9',
      parentId: '8',
      role: { type: 'role', value: 'generic' },
      name: str(''),
      childIds: ['26'],
    };
    const text = { ...staticText('26', '123-45-6789', '9'), childIds: ['-3'] };
    const inline = {
      ...staticText('-3', '123-45-6789', '26'),
      role: { type: 'role', value: 'InlineTextBox' },
    };
    const select = {
      nodeId: '13',
      role: { type: 'role', value: 'combobox' },
      name: str('PIN'),
      value: { type: 'string', value: '4321' },
      childIds: ['16'],
    };
    const popup = {
      nodeId: '16',
      parentId: '13',
      role: { type: 'role', value: 'MenuListPopup' },
      name: str(''),
      childIds: ['20'],
    };
    const option = {
      nodeId: '20',
      parentId: '16',
      role: { type: 'role', value: 'option' },
      name: str('4321'),
    };
    const out = await r.redactResult(
      AX,
      tree(input, generic, text, inline, select, popup, option),
    );
    expect(byId(out, '26').name?.value).toBe(MASK);
    expect(byId(out, '-3').name?.value).toBe(MASK);
    expect(byId(out, '20').name?.value).toBe(MASK);
    expect(byId(out, '9').name?.value).toBe('');
    expect(JSON.stringify(out)).not.toContain('6789');
    expect(JSON.stringify(out)).not.toContain('4321');
  });

  it('leaves an ordinary node untouched', async () => {
    const r = createRedactor();
    const input = tree(
      textbox('1', 'First name', 'Hopper', { childIds: ['2'] }),
      staticText('2', 'Hopper', '1'),
    );
    const out = await r.redactResult(AX, input);
    expect(out).toEqual(input);
  });

  it('does not classify or mask label text nodes that merely name a category', async () => {
    const r = createRedactor();
    const out = await r.redactResult(
      AX,
      tree(staticText('1', 'Social Security Number', '0')),
    );
    expect(byId(out, '1').name?.value).toBe('Social Security Number');
  });

  it('empty sensitive field stays empty', async () => {
    const r = createRedactor();
    const out = byId(
      await r.redactResult(AX, tree(textbox('1', 'SSN', ''))),
      '1',
    );
    expect(out.value?.value).toBe('');
    expect(out.name?.value).toBe(MASK);
  });

  it('a password-type node with an innocuous label is masked using describe()', async () => {
    const r = createRedactor();
    const node = textbox('6', 'Code ', '••••••••••', { backendDOMNodeId: 6 });
    const asked: number[] = [];
    const out = await r.redactResult(AX, tree(node), async (id) => {
      asked.push(id);
      return { inputType: 'password' };
    });
    expect(asked).toEqual([6]);
    expect(byId(out, '6').value?.value).toBe(MASK);
  });

  it('uses the placeholder in name.sources without describe()', async () => {
    const r = createRedactor();
    const node = textbox('1', 'Number', '111-22-3333', {
      name: {
        type: 'computedString',
        value: 'Number',
        sources: [
          {
            type: 'placeholder',
            attribute: 'placeholder',
            attributeValue: { type: 'string', value: 'xxx-xx-xxxx' },
            superseded: true,
          },
        ],
      },
    });
    const out = await r.redactResult(AX, tree(node));
    expect(byId(out, '1').value?.value).toBe(MASK);
  });

  it('an unknown node (describe returns undefined) with a sensitive label is masked', async () => {
    const r = createRedactor();
    const node = textbox('1', 'Bank account number', '000123456', {
      backendDOMNodeId: 1,
    });
    const out = await r.redactResult(AX, tree(node), async () => undefined);
    expect(byId(out, '1').value?.value).toBe(MASK);
  });

  it('describe() rejecting or throwing falls back to AX-only classification for that node', async () => {
    const r = createRedactor();
    const input = tree(
      textbox('1', 'SSN', '123-45-6789', { backendDOMNodeId: 1 }),
      // Residual risk, by decision: an input-like node whose type is unknown
      // and whose AX data carries no sensitive signal stays ordinary.
      textbox('2', 'Code', 'Hopper', { backendDOMNodeId: 2 }),
    );
    const rejecting = await r.redactResult(AX, input, (id) =>
      id === 1 ? Promise.reject(new Error('gone')) : Promise.reject('x'),
    );
    expect(byId(rejecting, '1').value?.value).toBe(MASK);
    expect(byId(rejecting, '2').value?.value).toBe('Hopper');
    const throwing = await r.redactResult(AX, input, () => {
      throw new Error('sync');
    });
    expect(byId(throwing, '1').value?.value).toBe(MASK);
    expect(byId(throwing, '2').value?.value).toBe('Hopper');
  });

  it('scrubs a known value and its digits-only, hyphenated and spaced variants from any string in any result or event, deeply', async () => {
    const r = createRedactor();
    expect(r.addKnownValue('123-45-6789')).toBe(true);
    const result = {
      result: {
        type: 'string',
        value: 'review: 123456789 and 123 45 6789',
      },
      nested: [{ deeper: ['x 123-45-6789 y'] }],
      frameTree: { frame: { url: 'https://e.test/p#ssn=123%2045%206789' } },
    };
    const out = JSON.stringify(
      await r.redactResult('Runtime.evaluate', result),
    );
    expect(out).not.toMatch(/6789/);
    expect(out).toContain(`review: ${MASK} and ${MASK}`);

    const event = r.redactEvent('Runtime.consoleAPICalled', {
      type: 'log',
      args: [{ type: 'string', value: 'SSN is 123.45.6789' }],
    });
    expect(JSON.stringify(event)).not.toMatch(/6789/);
    const dialog = r.redactEvent('Page.javascriptDialogOpening', {
      message: 'Confirm 123456789?',
    });
    expect(dialog).toEqual({ message: `Confirm ${MASK}?` });
  });

  it('scrubs known values case-insensitively and escapes regex metacharacters', async () => {
    const r = createRedactor();
    r.addKnownValue('a+b(c)');
    r.addKnownValue('Secret Answer');
    const out = await r.redactResult('Runtime.evaluate', {
      value: 'A+B(C) aab(c) SECRET ANSWER',
    });
    expect(out).toEqual({ value: `${MASK} aab(c) ${MASK}` });
  });

  it('scrubs known values from an AX tree too, outside sensitive nodes', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const out = await r.redactResult(
      AX,
      tree(staticText('1', 'You entered 123456789', '0')),
    );
    expect(byId(out, '1').name?.value).toBe(`You entered ${MASK}`);
  });

  it('masks the value attribute of a sensitive node in DOM.describeNode and scrubs other attributes', async () => {
    const r = createRedactor();
    r.addKnownValue('Hopper99');
    const out = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeName: 'FORM',
        nodeType: 1,
        attributes: ['aria-label', 'Login'],
        children: [
          {
            nodeName: 'INPUT',
            nodeType: 1,
            attributes: [
              'type',
              'password',
              'aria-label',
              'Code',
              'value',
              'fakepw1234',
            ],
          },
          {
            nodeName: 'INPUT',
            nodeType: 1,
            attributes: ['id', 'n', 'value', 'Hopper', 'title', 'Hopper99'],
          },
        ],
      },
    })) as { node: { children: Array<{ attributes: string[] }> } };
    expect(out.node.children[0].attributes).toEqual([
      'type',
      'password',
      'aria-label',
      'Code',
      'value',
      MASK,
    ]);
    expect(out.node.children[1].attributes).toEqual([
      'id',
      'n',
      'value',
      'Hopper',
      'title',
      MASK,
    ]);
  });

  it('masks text children of a sensitive DOM node (textarea content)', async () => {
    const r = createRedactor();
    const out = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeName: 'TEXTAREA',
        nodeType: 1,
        attributes: ['name', 'ssn'],
        children: [{ nodeName: '#text', nodeType: 3, nodeValue: '987654321' }],
      },
    })) as { node: { children: Array<{ nodeValue: string }> } };
    expect(out.node.children[0].nodeValue).toBe(MASK);
  });

  it('leaves screenshot data untouched', async () => {
    const r = createRedactor();
    r.addKnownValue('AAAABBBB');
    const out = await r.redactResult('Page.captureScreenshot', {
      data: 'xxAAAABBBBxx',
    });
    expect(out).toEqual({ data: 'xxAAAABBBBxx' });
  });

  it('ignores values shorter than 4 characters', async () => {
    const r = createRedactor();
    expect(r.addKnownValue('123')).toBe(true);
    const out = await r.redactResult('Runtime.evaluate', { value: 'abc 123' });
    expect(out).toEqual({ value: 'abc 123' });
  });

  it('fails closed beyond maxKnownValues and keeps earlier values', async () => {
    const r = createRedactor({ maxKnownValues: 2 });
    expect(r.addKnownValue('first-1111')).toBe(true);
    expect(r.addKnownValue('second-2222')).toBe(true);
    expect(r.addKnownValue('first-1111')).toBe(true); // already known
    expect(r.addKnownValue('third-3333')).toBe(false);
    const out = await r.redactResult('Runtime.evaluate', {
      value: 'first-1111 second-2222',
    });
    expect(out).toEqual({ value: `${MASK} ${MASK}` });
    expect(createRedactor().addKnownValue(42 as unknown as string)).toBe(false);
  });

  it('known values are cleared by clear()', async () => {
    const r = createRedactor({ maxKnownValues: 1 });
    r.addKnownValue('123-45-6789');
    r.clear();
    const out = await r.redactResult('Runtime.evaluate', {
      value: '123-45-6789',
    });
    expect(out).toEqual({ value: '123-45-6789' });
    expect(
      r.redactEvent('Page.javascriptDialogOpening', { m: '123456789' }),
    ).toEqual({ m: '123456789' });
    // The cap is freed too.
    expect(r.addKnownValue('other-9999')).toBe(true);
  });

  it('never mutates its input', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const input = tree(
      textbox('1', 'SSN', '123-45-6789', { childIds: ['2'] }),
      staticText('2', '123-45-6789', '1'),
    );
    const before = structuredClone(input);
    const out = await r.redactResult(AX, input);
    expect(input).toEqual(before);
    expect(out).not.toBe(input);
    const event = { message: '123456789' };
    r.redactEvent('Page.javascriptDialogOpening', event);
    expect(event).toEqual({ message: '123456789' });
  });

  it('redactResult never throws on a malformed result and returns a fully masked or empty structure instead', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const hostile = {
      get nodes(): unknown {
        throw new Error('boom 123-45-6789');
      },
    };
    await expect(r.redactResult(AX, hostile)).resolves.toEqual({ nodes: [] });
    await expect(r.redactResult(AX, { nodes: 'nope' })).resolves.toEqual({
      nodes: [],
    });
    await expect(r.redactResult(AX, null)).resolves.toEqual({ nodes: [] });
    await expect(
      r.redactResult(
        AX,
        tree(null as unknown as AXNode, 7 as unknown as AXNode),
      ),
    ).resolves.toEqual({ nodes: [] });
    const evil = {
      get value(): unknown {
        throw new Error('boom');
      },
    };
    await expect(r.redactResult('Runtime.evaluate', evil)).resolves.toBeNull();
    expect(r.redactEvent('Runtime.consoleAPICalled', evil)).toBeNull();
    // A cycle cannot leak either.
    const cyclic: Record<string, unknown> = { value: '123-45-6789' };
    cyclic.self = cyclic;
    expect(r.redactEvent('Runtime.consoleAPICalled', cyclic)).toBeNull();
  });

  it('scrubs a 10,000-node AX tree quickly', async () => {
    const r = createRedactor();
    for (let i = 0; i < 32; i++) r.addKnownValue(`fake-${1000 + i}-value`);
    const nodes: AXNode[] = [];
    for (let i = 0; i < 10_000; i++) {
      nodes.push(
        i % 10 === 0
          ? textbox(String(i), `Field ${i}`, `value ${i}`, {
              backendDOMNodeId: i,
            })
          : staticText(String(i), `Some ordinary page text number ${i}`, '0'),
      );
    }
    const start = performance.now();
    const out = await r.redactResult(AX, { nodes }, async () => ({
      inputType: 'text',
    }));
    const elapsed = performance.now() - start;
    expect(nodesOf(out)).toHaveLength(10_000);
    expect(elapsed).toBeLessThan(500);
  });
});

describe('descriptorFromAttributes', () => {
  it('maps DOM attributes to a field descriptor', () => {
    const d: FieldDescriptor = descriptorFromAttributes([
      'type',
      'Password',
      'autocomplete',
      'current-password',
      'name',
      'pw',
      'placeholder',
      'Code',
      'aria-label',
      'Code',
      'aria-description',
      'six digits',
      'value',
      'ignored',
    ]);
    expect(d).toEqual({
      inputType: 'Password',
      autocomplete: 'current-password',
      name: 'pw',
      placeholder: 'Code',
      label: 'Code',
      description: 'six digits',
    });
  });
});
