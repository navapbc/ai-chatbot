import { describe, expect, it } from 'vitest';
import {
  createRedactor,
  descriptorFromAttributes,
  hasSensitiveShape,
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

  it('drops variants shorter than 3 characters', () => {
    const v = valueVariants('a-b');
    expect(v.every((x) => x.length >= 3)).toBe(true);
    expect(v).not.toContain('ab');
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
      textbox('1', 'First name', 'Hopper', {
        childIds: ['2'],
        backendDOMNodeId: 1,
      }),
      staticText('2', 'Hopper', '1'),
    );
    const out = await r.redactResult(AX, input, async () => ({
      inputType: 'text',
    }));
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
    // A text input the page masks with CSS: AX shows the clear value.
    const node = textbox('6', 'Code ', 'hunter22', { backendDOMNodeId: 6 });
    const asked: number[] = [];
    const out = await r.redactResult(AX, tree(node), async (id) => {
      asked.push(id);
      return { inputType: 'password' };
    });
    expect(asked).toEqual([6]);
    expect(byId(out, '6').value?.value).toBe(MASK);
  });

  it('uses the placeholder in name.sources when describe() finds no signal', async () => {
    const r = createRedactor();
    const node = textbox('1', 'Number', '111-22-3333', {
      backendDOMNodeId: 1,
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
    const out = await r.redactResult(AX, tree(node), async () => ({
      inputType: 'text',
    }));
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

  it('describe() rejecting or throwing fails closed: a filled field-like node is masked', async () => {
    const r = createRedactor();
    const input = tree(
      textbox('1', 'SSN', '123-45-6789', { backendDOMNodeId: 1 }),
      // No AX signal and the type is unknown: fail toward sensitive.
      textbox('2', 'Code', 'Hopper', { backendDOMNodeId: 2 }),
      // Empty stays empty and keeps its label.
      textbox('3', 'Code', '', { backendDOMNodeId: 3 }),
    );
    const rejecting = await r.redactResult(AX, input, (id) =>
      id === 1 ? Promise.reject(new Error('gone')) : Promise.reject('x'),
    );
    expect(byId(rejecting, '1').value?.value).toBe(MASK);
    expect(byId(rejecting, '2').value?.value).toBe(MASK);
    expect(byId(rejecting, '3').value?.value).toBe('');
    expect(byId(rejecting, '3').name?.value).toBe('Code');
    const throwing = await r.redactResult(AX, input, () => {
      throw new Error('sync');
    });
    expect(byId(throwing, '1').value?.value).toBe(MASK);
    expect(byId(throwing, '2').value?.value).toBe(MASK);
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
    // The value attribute of every INPUT is masked, sensitive or not.
    expect(out.node.children[0].attributes).toEqual([
      'type',
      'password',
      'aria-label',
      MASK,
      'value',
      MASK,
    ]);
    expect(out.node.children[1].attributes).toEqual([
      'id',
      'n',
      'value',
      MASK,
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

  it('ignores values shorter than 3 characters and scrubs 3-character values as whole tokens only', async () => {
    const r = createRedactor();
    expect(r.addKnownValue('12')).toBe(true);
    expect(r.addKnownValue('Abc')).toBe(true);
    const out = await r.redactResult('Runtime.evaluate', {
      value: 'abc 12 abcd xabc ABC.',
    });
    expect(out).toEqual({ value: `${MASK} 12 abcd xabc ${MASK}.` });
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
    const out = await r.redactResult('Page.getNavigationHistory', {
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

describe('review regressions', () => {
  const text = (node: unknown) =>
    (node as { name?: { value?: unknown } }).name?.value;
  const ordinary = async () => ({ inputType: 'text' });

  it('C1: masks the value attribute of an INPUT classified only by its label', async () => {
    const r = createRedactor();
    const out = await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        attributes: [
          'id',
          'f1',
          'name',
          'field_17',
          'type',
          'text',
          'value',
          '123-45-6789',
        ],
      },
    });
    expect(JSON.stringify(out)).not.toContain('6789');
    expect((out as { node: { attributes: string[] } }).node.attributes).toEqual(
      ['id', 'f1', 'name', 'field_17', 'type', 'text', 'value', MASK],
    );
  });

  it('C1: masks TEXTAREA text and SELECT/OPTION values and text unconditionally', async () => {
    const r = createRedactor();
    const out = JSON.stringify(
      await r.redactResult('DOM.describeNode', {
        node: {
          nodeType: 1,
          nodeName: 'DIV',
          attributes: [],
          children: [
            {
              nodeType: 1,
              nodeName: 'TEXTAREA',
              attributes: ['id', 'notes'],
              children: [
                { nodeType: 3, nodeName: '#text', nodeValue: '111-22-3333' },
              ],
            },
            {
              nodeType: 1,
              nodeName: 'SELECT',
              attributes: ['id', 's'],
              children: [
                {
                  nodeType: 1,
                  nodeName: 'OPTION',
                  attributes: ['value', '987654321', 'label', '987654321'],
                  children: [
                    { nodeType: 3, nodeName: '#text', nodeValue: '987654321' },
                  ],
                },
              ],
            },
            { nodeType: 3, nodeName: '#text', nodeValue: 'Ordinary text' },
          ],
        },
      }),
    );
    expect(out).not.toContain('3333');
    expect(out).not.toContain('987654321');
    expect(out).toContain('Ordinary text');
  });

  it('M3: a sensitive DOM node keeps only type, name, id, autocomplete and class', async () => {
    const r = createRedactor();
    const out = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        attributes: [
          'type',
          'text',
          'name',
          'ssn',
          'id',
          'ssn',
          'autocomplete',
          'off',
          'class',
          'field',
          'aria-label',
          'SSN',
          'title',
          '123-45-6789',
          'data-prefill',
          '123-45-6789',
          'placeholder',
          '123-45-6789',
          'value',
          '123-45-6789',
        ],
      },
    })) as { node: { attributes: string[] } };
    expect(out.node.attributes).toEqual([
      'type',
      'text',
      'name',
      'ssn',
      'id',
      'ssn',
      'autocomplete',
      'off',
      'class',
      'field',
      'aria-label',
      MASK,
      'title',
      MASK,
      'data-prefill',
      MASK,
      'placeholder',
      MASK,
      'value',
      MASK,
    ]);
  });

  it('I1: a known value split across a StaticText and its InlineTextBoxes is masked', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const out = await r.redactResult(
      AX,
      tree(
        {
          ...staticText('1', 'Your SSN is 123-45-6789 on file', '0'),
          childIds: ['2', '3'],
        },
        {
          ...staticText('2', 'Your SSN is 123-45-', '1'),
          role: { type: 'role', value: 'InlineTextBox' },
        },
        {
          ...staticText('3', '6789 on file', '1'),
          role: { type: 'role', value: 'InlineTextBox' },
        },
      ),
    );
    expect(text(byId(out, '1'))).toBe(`Your SSN is ${MASK} on file`);
    expect(JSON.stringify(out)).not.toMatch(/6789|123-45/);
  });

  it('I1: a known value split across sibling StaticText nodes is masked, other text kept', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const out = await r.redactResult(
      AX,
      tree(
        {
          nodeId: '0',
          role: { type: 'role', value: 'paragraph' },
          name: str(''),
          childIds: ['1', '2', '3', '4', '5'],
        },
        staticText('1', 'Hello ', '0'),
        staticText('2', '123', '0'),
        staticText('3', '-45-', '0'),
        staticText('4', '6789', '0'),
        staticText('5', ' thanks', '0'),
      ),
    );
    expect(text(byId(out, '1'))).toBe('Hello ');
    expect(text(byId(out, '5'))).toBe(' thanks');
    for (const id of ['2', '3', '4']) expect(text(byId(out, id))).toBe(MASK);
  });

  it('I2: well-formed protocol identifiers are not scrubbed; URL hosts are', async () => {
    const r = createRedactor();
    r.addKnownValue('1234');
    r.addKnownValue('Example');
    const hex = '0123ABCD1234EF5678901234ABCDEF12';
    const ids = {
      sessionId: hex,
      targetId: hex,
      frameId: hex,
      loaderId: hex,
      browserContextId: hex,
      objectId: '-781234.1.1234',
      requestId: '1234.56',
      uniqueId: '-1234.1234',
      scriptId: '1234',
      executionContextId: 1234,
    };
    const out = await r.redactResult('Target.attachToTarget', {
      ...ids,
      url: 'https://example.test/path/Example?q=1234#Example',
      text: 'PIN 1234 for Example',
    });
    expect(out).toEqual({
      ...ids,
      url: `https://${MASK}.test/path/${MASK}?q=${MASK}#${MASK}`,
      text: `PIN ${MASK} for ${MASK}`,
    });
    const ax = await r.redactResult(
      AX,
      tree(
        {
          nodeId: '1234',
          role: { type: 'role', value: 'paragraph' },
          name: str(''),
          childIds: ['12345', '-1234'],
          properties: [{ name: 'level', value: { type: 'integer', value: 2 } }],
          backendDOMNodeId: 1234,
          frameId: hex,
        },
        staticText('12345', 'plain', '1234'),
      ),
    );
    const root = byId(ax, '1234');
    expect(root.childIds).toEqual(['12345', '-1234']);
    expect(root.frameId).toBe(hex);
    expect(root.properties?.[0]).toEqual({
      name: 'level',
      value: { type: 'integer', value: 2 },
    });
    expect(text(byId(ax, '12345'))).toBe('plain');
  });

  it('I2: Page.getFrameTree frame ids survive a known value', async () => {
    const r = createRedactor();
    r.addKnownValue('1234');
    const id = 'ABCDEF0123456789AB1234CDEF012345';
    const out = (await r.redactResult('Page.getFrameTree', {
      frameTree: {
        frame: {
          id,
          loaderId: id,
          url: 'https://a.test/?pin=1234',
          securityOrigin: 'https://a.test',
          mimeType: 'text/html',
        },
        childFrames: [
          { frame: { id, parentId: id, loaderId: id, url: 'about:blank' } },
        ],
      },
    })) as {
      frameTree: {
        frame: Record<string, string>;
        childFrames: Array<{ frame: Record<string, string> }>;
      };
    };
    expect(out.frameTree.frame.id).toBe(id);
    expect(out.frameTree.frame.loaderId).toBe(id);
    expect(out.frameTree.frame.url).toBe(`https://a.test/?pin=${MASK}`);
    expect(out.frameTree.childFrames[0].frame.parentId).toBe(id);
  });

  it('I2: the structural exemption does not cover page-controlled or malformed positions', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const out = JSON.stringify(
      await r.redactResult('DOM.describeNode', {
        sessionId: 'x 123-45-6789',
        frameId: '123-45-6789',
        type: '123-45-6789',
        node: {
          nodeType: 1,
          nodeName: 'X-123-45-6789',
          localName: 'x-123-45-6789',
          attributes: ['data-123-45-6789', ''],
        },
        doctype: { nodeType: 10, nodeName: '123-45-6789' },
        ax: {
          role: { type: 'role', value: '123-45-6789' },
          properties: [
            { name: 'x123-45-6789', value: { type: 'token', value: 'a' } },
          ],
        },
      }),
    );
    expect(out).not.toMatch(/6789/);
  });

  it('I2: page data returned by value gets no structural exemption', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    const out = await r.redactResult('Runtime.callFunctionOn', {
      result: {
        type: 'object',
        value: {
          type: '123-45-6789',
          nodeId: '123-45-6789',
          sessionId: ['123456789'],
        },
      },
    });
    expect(JSON.stringify(out)).not.toMatch(/6789/);
    expect((out as { result: { type: string } }).result.type).toBe('object');
  });

  it('I2: a URL with userinfo is scrubbed whole', async () => {
    const r = createRedactor();
    r.addKnownValue('hunter22');
    const out = await r.redactResult('Page.getFrameTree', {
      url: 'https://me:hunter22@host.test/',
    });
    expect(JSON.stringify(out)).not.toContain('hunter22');
  });

  it('I3: describe() hanging times out and the filled node is masked', async () => {
    const r = createRedactor({ describeTimeoutMs: 20 });
    const out = await r.redactResult(
      AX,
      tree(textbox('1', 'Code', 'Hopper', { backendDOMNodeId: 1 })),
      () => new Promise(() => {}),
    );
    expect(byId(out, '1').value?.value).toBe(MASK);
  });

  it('I3: describe() omitted, or no backendDOMNodeId, masks a filled field-like node', async () => {
    const r = createRedactor();
    const omitted = await r.redactResult(
      AX,
      tree(textbox('1', 'Code', 'Hopper', { backendDOMNodeId: 1 })),
    );
    expect(byId(omitted, '1').value?.value).toBe(MASK);
    const noId = await r.redactResult(
      AX,
      tree(textbox('1', 'Code', 'Hopper')),
      ordinary,
    );
    expect(byId(noId, '1').value?.value).toBe(MASK);
  });

  it('I3: a bullets-only value is sensitive whatever describe() says', async () => {
    const r = createRedactor();
    for (const bullets of ['••••••', '●●●●', '****', '∗∗∗∗']) {
      const out = await r.redactResult(
        AX,
        tree(textbox('1', 'Code', bullets, { backendDOMNodeId: 1 })),
        ordinary,
      );
      expect(byId(out, '1').value?.value).toBe(MASK);
    }
  });

  it('I3: the AX autocomplete property (aria-autocomplete) is not a signal', async () => {
    const r = createRedactor();
    const node = textbox('1', 'Name', 'Hopper', {
      backendDOMNodeId: 1,
      properties: [
        { name: 'autocomplete', value: { type: 'token', value: 'list' } },
      ],
    });
    const out = await r.redactResult(AX, tree(node), ordinary);
    expect(byId(out, '1').value?.value).toBe('Hopper');
  });

  it('I4: partial AX payloads fail closed', async () => {
    const r = createRedactor();
    const payload = tree(staticText('1', 'Hello', '0'));
    for (const method of [
      'Accessibility.getChildAXNodes',
      'Accessibility.getPartialAXTree',
      'Accessibility.queryAXTree',
    ]) {
      await expect(r.redactResult(method, payload)).resolves.toEqual({
        nodes: [],
      });
    }
    expect(r.redactEvent('Accessibility.nodesUpdated', payload)).toBeNull();
  });

  it('I4: Accessibility.loadComplete masks its root like a tree', () => {
    const r = createRedactor();
    const out = r.redactEvent('Accessibility.loadComplete', {
      root: textbox('1', 'Code', 'Hopper'),
    }) as { root: { value: { value: unknown } } };
    expect(out.root.value.value).toBe(MASK);
  });

  it('I4: DOM payloads whose owning node cannot be classified are dropped or masked', async () => {
    const r = createRedactor();
    await expect(
      r.redactResult('DOM.getOuterHTML', {
        outerHTML: '<input value="123-45-6789">',
      }),
    ).resolves.toBeNull();
    await expect(
      r.redactResult('DOM.getAttributes', {
        attributes: ['id', 'x', 'value', '123-45-6789'],
      }),
    ).resolves.toEqual({ attributes: ['id', 'x', 'value', MASK] });
    for (const [method, params] of [
      ['DOM.attributeModified', { nodeId: 1, name: 'value', value: '1234' }],
      ['DOM.characterDataModified', { nodeId: 1, characterData: '1234' }],
      ['DOM.setChildNodes', { parentId: 1, nodes: [] }],
      ['DOM.childNodeInserted', { parentNodeId: 1, node: {} }],
    ] as const) {
      expect(r.redactEvent(method, params)).toBeNull();
    }
  });

  it('I5: digit values are scrubbed across unusual separators, widths and encodings', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    for (const form of [
      '123\u00a045\u00a06789',
      '123\u201345\u20136789',
      '123\u200b45-67\u200b89',
      '123\u00ad45\u00ad6789',
      '１２３-４５-６７８９',
      'ssn=123-45-6789&x=1',
      'ssn=123%2D45%2D6789',
      'a%20123456789',
      '{"ssn":"123\\u002d45\\u002d6789"}',
    ]) {
      const out = JSON.stringify(
        await r.redactResult('Runtime.evaluate', { value: form }),
      );
      expect(out, form).toContain(MASK);
      expect(out, form).not.toMatch(/6789|６７８９/);
    }
    // A longer number is not the value (no shape guard on this method).
    expect(
      await r.redactResult('Page.getNavigationHistory', {
        value: '1234567890',
      }),
    ).toEqual({ value: '1234567890' });
  });

  it('I5: text values are scrubbed in URL, form, JSON and HTML encodings and ß forms', async () => {
    const r = createRedactor();
    r.addKnownValue('Secret Answer!');
    r.addKnownValue('Pa"ss\\word');
    r.addKnownValue('Tom & Jerry <3');
    r.addKnownValue('Straße');
    for (const form of [
      'q=Secret%20Answer!',
      'q=Secret+Answer%21',
      '"Pa\\"ss\\\\word"',
      '<p>Tom &amp; Jerry &lt;3</p>',
      'STRASSE',
      'STRAẞE',
      'Strasse',
    ]) {
      const out = (await r.redactResult('Runtime.evaluate', {
        value: form,
      })) as { value: string };
      expect(out.value, form).toContain(MASK);
      expect(out.value, form).not.toMatch(/Secret|ss\\\\w|Jerry|stra/i);
    }
  });

  it('I6: a node labelled or described by a sensitive node is masked', async () => {
    const r = createRedactor();
    const input = textbox('1', 'SSN', '123-45-6789', { backendDOMNodeId: 50 });
    const related = (id: number) => ({
      type: 'nodeList',
      relatedNodes: [{ backendDOMNodeId: id, text: '123-45-6789' }],
    });
    const button = {
      nodeId: '2',
      role: { type: 'role', value: 'button' },
      name: str('Confirm 123-45-6789'),
      properties: [{ name: 'labelledby', value: related(50) }],
    };
    const other = {
      nodeId: '3',
      role: { type: 'role', value: 'button' },
      name: str('Back'),
      properties: [{ name: 'describedby', value: related(99) }],
    };
    const out = await r.redactResult(AX, tree(input, button, other), ordinary);
    expect(text(byId(out, '2'))).toBe(MASK);
    expect(JSON.stringify(byId(out, '2'))).not.toContain('6789');
    expect(text(byId(out, '3'))).toBe('Back');
  });

  it('M5: a sensitive numeric AX value stays a well-typed AXValue', async () => {
    const r = createRedactor();
    const out = await r.redactResult(
      AX,
      tree({
        nodeId: '1',
        role: { type: 'role', value: 'spinbutton' },
        name: str('PIN'),
        value: { type: 'number', value: 4321 },
      }),
    );
    expect(byId(out, '1').value).toEqual({ type: 'string', value: MASK });
  });

  it('M5: refuses known values longer than 512 characters', () => {
    const r = createRedactor();
    expect(r.addKnownValue('x'.repeat(512))).toBe(true);
    expect(r.addKnownValue('y'.repeat(513))).toBe(false);
  });

  it('bounds describe() calls on a 10,000-node tree with a 5 ms describe', async () => {
    const r = createRedactor();
    const nodes: AXNode[] = [];
    for (let i = 0; i < 10_000; i++) {
      nodes.push(
        i % 10 === 0
          ? textbox(String(i), `Field ${i}`, `value ${i}`, {
              backendDOMNodeId: i % 20 === 0 ? i : i - 10, // duplicate ids
            })
          : staticText(String(i), `Some ordinary page text number ${i}`, '0'),
      );
    }
    let calls = 0;
    let active = 0;
    let peak = 0;
    const start = performance.now();
    await r.redactResult(AX, { nodes }, async () => {
      calls++;
      active++;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active--;
      return { inputType: 'text' };
    });
    expect(performance.now() - start).toBeLessThan(500);
    expect(peak).toBeLessThanOrEqual(16);
    // One call per distinct backendDOMNodeId of a filled field-like node.
    expect(calls).toBe(500);
  });
});

describe('review regressions, round 2', () => {
  const text = (node: unknown) =>
    (node as { name?: { value?: unknown } }).name?.value;
  const ordinary = async () => ({ inputType: 'text' });

  it('C1: a URL followed by text, or a value inside a host, is scrubbed', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    for (const form of [
      'https://x 123-45-6789/path',
      'https://a\\123-45-6789/',
      'x://123-45-6789',
      'https://123-45-6789.evil.test/',
      'https://evil.test:123456789/',
    ]) {
      const out = JSON.stringify(
        await r.redactResult('Runtime.evaluate', { result: { value: form } }),
      );
      expect(out, form).not.toMatch(/6789/);
    }
    const ax = await r.redactResult(
      AX,
      tree(
        staticText('1', 'https://benefits.test SSN 123-45-6789', '0'),
        textbox('2', 'Website', 'http://a 123-45-6789', {
          backendDOMNodeId: 2,
        }),
      ),
      ordinary,
    );
    expect(JSON.stringify(ax)).not.toMatch(/6789/);
    const event = r.redactEvent('Network.requestWillBeSent', {
      requestId: '1.1',
      request: { url: 'https://h.test 123-45-6789', method: 'GET' },
    });
    expect(JSON.stringify(event)).not.toMatch(/6789/);
  });

  it('raw-first: a known value equal to a structural word does not disable DOM masking', async () => {
    const r = createRedactor();
    r.addKnownValue('password');
    r.addKnownValue('input');
    const out = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        localName: 'input',
        attributes: ['type', 'password', 'title', 'hint', 'value', 'abcdef'],
      },
    })) as { node: { attributes: string[] } };
    // type=password still classifies it sensitive (title masked by the
    // allowlist), and INPUT still gets its value masked.
    expect(out.node.attributes[3]).toBe(MASK);
    expect(out.node.attributes[5]).toBe(MASK);
    const plain = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        attributes: ['type', 'text', 'value', '123-45-6789'],
      },
    })) as { node: { attributes: string[] } };
    expect(plain.node.attributes[3]).toBe(MASK);
  });

  it('raw-first: a known value equal to a label word does not hide the AX signal', async () => {
    const r = createRedactor();
    r.addKnownValue('SSN');
    const out = await r.redactResult(
      AX,
      tree(textbox('1', 'SSN', 'Hopper99', { backendDOMNodeId: 1 })),
      ordinary,
    );
    expect(byId(out, '1').value?.value).toBe(MASK);
    const dom = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        attributes: ['name', 'ssn', 'title', 'hint'],
      },
    })) as { node: { attributes: string[] } };
    expect(dom.node.attributes[3]).toBe(MASK);
  });

  it('I2: descriptorFromAttributes keeps id alongside name, and title', async () => {
    const d = descriptorFromAttributes([
      'name',
      'field_17',
      'id',
      'ssn',
      'title',
      'Routing number',
    ]);
    expect(d.name).toContain('field_17');
    expect(d.name).toContain('ssn');
    expect(d.description).toContain('Routing number');
    const r = createRedactor();
    const out = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        attributes: ['name', 'field_17', 'id', 'ssn', 'data-x', 'y'],
      },
    })) as { node: { attributes: string[] } };
    expect(out.node.attributes[5]).toBe(MASK);
  });

  it('I3: a known value split by markup is masked', async () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    for (const html of [
      '<span>123</span>-<span>45</span>-<span>6789</span>',
      '<td>123</td><td>45</td><td>6789</td>',
      '123<b>45</b>6789',
    ]) {
      for (const method of ['Runtime.callFunctionOn', 'Runtime.evaluate']) {
        const out = await r.redactResult(method, {
          result: { type: 'string', value: html },
        });
        expect(out, html).toEqual({ result: { type: 'string', value: MASK } });
      }
    }
    expect(
      await r.redactResult('Runtime.evaluate', { value: '<b>Hello</b>' }),
    ).toEqual({ value: '<b>Hello</b>' });
  });

  it('an image input value is masked (only text-labelled buttons keep it)', async () => {
    const r = createRedactor();
    const out = (await r.redactResult('DOM.describeNode', {
      node: {
        nodeType: 1,
        nodeName: 'INPUT',
        attributes: ['type', 'image', 'value', 'Go'],
      },
    })) as { node: { attributes: string[] } };
    expect(out.node.attributes[3]).toBe(MASK);
  });

  it('a hanging describe() on a 10,000-node tree is bounded by the overall deadline', async () => {
    const r = createRedactor({
      describeTimeoutMs: 20,
      describeDeadlineMs: 100,
    });
    const nodes: AXNode[] = [];
    for (let i = 0; i < 10_000; i++) {
      nodes.push(
        i % 10 === 0
          ? textbox(String(i), `Field ${i}`, `value ${i}`, {
              backendDOMNodeId: i,
            })
          : staticText(String(i), `text ${i}`, '0'),
      );
    }
    let started = 0;
    const start = performance.now();
    const out = await r.redactResult(AX, { nodes }, () => {
      started++;
      return new Promise(() => {});
    });
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(400);
    // 16 workers, each abandoning a call per 20 ms timeout, for 100 ms.
    expect(started).toBeLessThanOrEqual(16 * 6);
    // Every filled field that could not be vouched for is masked.
    for (let i = 0; i < 10_000; i += 10) {
      expect(byId(out, String(i)).value?.value).toBe(MASK);
    }
    expect(text(byId(out, '1'))).toBe('text 1');
  });
});

describe('review regressions, round 3', () => {
  const known = () => {
    const r = createRedactor();
    r.addKnownValue('123-45-6789');
    return r;
  };

  it('masks numbers in page data returned by value', async () => {
    const r = known();
    expect(
      await r.redactResult('Runtime.evaluate', {
        result: { type: 'number', value: 123456789 },
      }),
    ).toEqual({ result: { type: 'number', value: MASK } });
    expect(
      await r.redactResult('Runtime.callFunctionOn', {
        result: { type: 'object', value: { ssn: 123456789, n: [123456789] } },
      }),
    ).toEqual({
      result: { type: 'object', value: { ssn: MASK, n: [MASK] } },
    });
    // Ordinary numbers and trusted numeric fields are untouched.
    expect(
      await r.redactResult('Runtime.callFunctionOn', {
        executionContextId: 123456789,
        result: { type: 'object', value: { count: 42, small: 12345678 } },
      }),
    ).toEqual({
      executionContextId: 123456789,
      result: { type: 'object', value: { count: 42, small: 12345678 } },
    });
  });

  it('scrubs keys of page data and header maps, uniquely, but not protocol field names', async () => {
    const r = known();
    r.addKnownValue('url');
    r.addKnownValue('name');
    const out = await r.redactResult('Runtime.callFunctionOn', {
      result: {
        type: 'object',
        value: { 'ssn 123-45-6789': true, 'ssn 123456789': 2, ok: 1 },
      },
    });
    expect(out).toEqual({
      result: {
        type: 'object',
        value: { [`ssn ${MASK}`]: true, [`ssn ${MASK}#2`]: 2, ok: 1 },
      },
    });
    const event = r.redactEvent('Network.requestWillBeSent', {
      requestId: '1.1',
      request: {
        url: 'https://h.test/',
        method: 'GET',
        headers: { 'X-SSN-123-45-6789': 'a', Accept: 'text/html' },
      },
      name: 'kept',
    }) as { request: Record<string, unknown>; name: string };
    expect(JSON.stringify(event)).not.toMatch(/6789/);
    expect(Object.keys(event.request)).toEqual(['url', 'method', 'headers']);
    expect(event.request.headers).toEqual({
      [`X-SSN-${MASK}`]: 'a',
      Accept: 'text/html',
    });
    expect(event.name).toBe('kept');
  });

  it('keeps a __proto__ key of page data as data', async () => {
    const r = known();
    const out = (await r.redactResult('Runtime.evaluate', {
      result: {
        type: 'object',
        value: JSON.parse('{"__proto__": "123-45-6789"}'),
      },
    })) as { result: { value: object } };
    expect(JSON.stringify(out)).not.toMatch(/6789/);
    expect(Object.getPrototypeOf(out.result.value)).toBe(Object.prototype);
    expect(Object.keys(out.result.value)).toEqual(['__proto__']);
  });

  it('masks a value split by pretty-printed markup, comments, entities or a lone <', async () => {
    const r = known();
    for (const form of [
      '<table>\n        <td>123</td>\n        <td>45</td>\n        <td>6789</td>',
      '123<!-- a > b -->45-6789',
      '123&lt;/b&gt;45&lt;b&gt;6789',
      '123<45-6789',
      '123\n                45-6789',
    ]) {
      const out = await r.redactResult('Runtime.evaluate', {
        result: { type: 'string', value: form },
      });
      expect(out, form).toEqual({ result: { type: 'string', value: MASK } });
    }
    expect(
      await r.redactResult('Runtime.evaluate', {
        value: '<p>\n    Hello &amp; welcome\n</p>',
      }),
    ).toEqual({ value: '<p>\n    Hello &amp; welcome\n</p>' });
  });
});

describe('review regressions, round 4', () => {
  const describeAttrs = async (attributes: string[], known?: string) => {
    const r = createRedactor();
    if (known) r.addKnownValue(known);
    const out = (await r.redactResult('DOM.describeNode', {
      node: { nodeType: 1, nodeName: 'INPUT', attributes },
    })) as { node: { attributes: string[] } };
    return Object.fromEntries(
      out.node.attributes.flatMap((v, i, a) => (i % 2 ? [] : [[v, a[i + 1]]])),
    );
  };

  it('masks every page-authored attribute of a sensitive DOM node', async () => {
    const attrs = await describeAttrs([
      'name',
      'bank_account',
      'aria-label',
      '900-55-0001',
      'aria-description',
      '900-55-0002',
      'aria-roledescription',
      '900-55-0003',
      'aria-placeholder',
      '900-55-0004',
      'aria-valuetext',
      '900-55-0005',
      'aria-valuenow',
      '900550006',
      'aria-labelledby',
      'lbl 900-55-0007',
      'aria-describedby',
      'd900-55-0008',
      'title',
      '900-55-0009',
      'role',
      '900-55-0010',
      'data-v',
      '900-55-0011',
      'required',
      '',
    ]);
    expect(JSON.stringify(attrs)).not.toMatch(/900-?55/);
    expect(attrs.name).toBe('bank_account');
    // Empty stays empty.
    expect(attrs.required).toBe('');
  });

  it('keeps type and autocomplete only when they are well-formed tokens', async () => {
    const good = await describeAttrs([
      'type',
      'password',
      'autocomplete',
      'section-blue shipping cc-number',
    ]);
    expect(good).toEqual({
      type: 'password',
      autocomplete: 'section-blue shipping cc-number',
    });
    const bad = await describeAttrs([
      'type',
      'password 900-55-0001',
      'autocomplete',
      '900-55-0001',
      'name',
      'pin',
    ]);
    expect(bad).toEqual({ type: MASK, autocomplete: MASK, name: 'pin' });
  });

  it('keeps name, id and class of a sensitive node but scrubs known values in them', async () => {
    const attrs = await describeAttrs(
      [
        'type',
        'password',
        'name',
        'pw-900-55-0001',
        'id',
        'x 900-55-0001',
        'class',
        'c 900550001',
      ],
      '900-55-0001',
    );
    expect(attrs).toEqual({
      type: 'password',
      name: `pw-${MASK}`,
      id: `x ${MASK}`,
      class: `c ${MASK}`,
    });
  });

  it('masks aria-derived text of a sensitive AX node', async () => {
    const r = createRedactor();
    const node = {
      nodeId: '1',
      role: { type: 'role', value: 'textbox' },
      name: {
        type: 'computedString',
        value: 'Bank account 900-55-0001',
        sources: [
          {
            type: 'attribute',
            attribute: 'aria-label',
            attributeValue: {
              type: 'string',
              value: 'Bank account 900-55-0001',
            },
          },
          {
            type: 'placeholder',
            attribute: 'aria-placeholder',
            attributeValue: { type: 'string', value: '900-55-0004' },
          },
        ],
      },
      description: { type: 'computedString', value: '900-55-0002' },
      value: { type: 'string', value: '' },
      properties: [
        {
          name: 'roledescription',
          value: { type: 'string', value: '900-55-0003' },
        },
        { name: 'valuetext', value: { type: 'string', value: '900-55-0005' } },
      ],
    };
    const out = await r.redactResult(AX, tree(node));
    expect(JSON.stringify(out)).not.toMatch(/900-55/);
    expect(byId(out, '1').value?.value).toBe('');
  });
});

describe('review regressions, round 5: value shapes (nothing registered)', () => {
  it('hasSensitiveShape: SSN-like, 9+ digit runs and cards, not decimals or short numbers', () => {
    for (const yes of [
      '900-00-1234',
      'SSN 900 00 1234 on file',
      '900.00.1234',
      '900001234',
      '１２３-４５-６７８９',
      'Order 1234567890',
      '4111 1111 1111 1111',
      '4111111111111112',
      '123456789.5',
    ]) {
      expect(hasSensitiveShape(yes), yes).toBe(true);
    }
    for (const no of [
      '12345678',
      '412.3333435058594',
      'Page 2 of 3',
      '(555) 123-4567',
      '2026-10-26',
      '',
    ]) {
      expect(hasSensitiveShape(no), no).toBe(false);
    }
  });

  it('masks sensitive query and fragment parameter values in URLs, keeps paths and hosts', () => {
    const r = createRedactor();
    const event = r.redactEvent('Network.requestWillBeSent', {
      requestId: '1.1',
      documentURL: 'https://b.test/apply?ssn=900-00-1234&page=2',
      request: {
        url: 'https://b.test/apply?x=900001234&sort=name#pin=4321&tab=2',
        method: 'POST',
        postData: 'ssn=900-00-1234&first=Ada',
        headers: {
          Referer: 'https://b.test/form?ssn=900-00-1234',
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      },
    }) as {
      documentURL: string;
      request: { url: string; postData: string; headers: { Referer: string } };
    };
    expect(event.documentURL).toBe(`https://b.test/apply?ssn=${MASK}&page=2`);
    expect(event.request.url).toBe(
      `https://b.test/apply?x=${MASK}&sort=name#pin=${MASK}&tab=2`,
    );
    expect(event.request.postData).toBe(`ssn=${MASK}&first=Ada`);
    expect(event.request.headers.Referer).toBe(
      `https://b.test/form?ssn=${MASK}`,
    );
  });

  it('leaves an ordinary URL unchanged', () => {
    const r = createRedactor();
    const url = 'https://b.test/list/2?page=2&sort=name#top';
    expect(r.redactEvent('Page.navigatedWithinDocument', { url })).toEqual({
      url,
    });
  });

  it('masks URL values in frame, navigation and target payloads', async () => {
    const r = createRedactor();
    const url = 'https://b.test/done?case_id=A12&v=900-00-1234';
    const masked = `https://b.test/done?case_id=${MASK}&v=${MASK}`;
    const frame = { id: 'ABCDEF0123456789AB1234CDEF012345', url };
    expect(r.redactEvent('Page.frameNavigated', { frame })).toEqual({
      frame: { ...frame, url: masked },
    });
    expect(
      await r.redactResult('Page.getFrameTree', { frameTree: { frame } }),
    ).toEqual({ frameTree: { frame: { ...frame, url: masked } } });
    expect(r.redactEvent('Page.navigatedWithinDocument', { url })).toEqual({
      url: masked,
    });
    for (const method of [
      'Target.targetInfoChanged',
      'Target.attachedToTarget',
    ]) {
      expect(
        JSON.stringify(
          r.redactEvent(method, { targetInfo: { url, title: 't' } }),
        ),
      ).not.toMatch(/1234|A12/);
    }
    const text = (await r.redactResult('Runtime.evaluate', {
      result: { type: 'string', value: 'see https://b.test/?pin=4321 now' },
    })) as { result: { value: string } };
    expect(text.result.value).toBe(`see https://b.test/?pin=${MASK} now`);
  });

  it('masks value-shaped script results: strings, by-value leaves, numbers, exceptions', async () => {
    const r = createRedactor();
    expect(
      await r.redactResult('Runtime.evaluate', {
        result: { type: 'string', value: '900-00-1234' },
      }),
    ).toEqual({ result: { type: 'string', value: MASK } });
    expect(
      await r.redactResult('Runtime.callFunctionOn', {
        result: { type: 'object', value: { a: { b: '900-00-1234', c: 'ok' } } },
      }),
    ).toEqual({
      result: { type: 'object', value: { a: { b: MASK, c: 'ok' } } },
    });
    expect(
      await r.redactResult('Runtime.evaluate', {
        result: { type: 'number', value: 900001234, description: '900001234' },
      }),
    ).toEqual({ result: { type: 'number', value: MASK, description: MASK } });
    for (const card of ['4111111111111111', '4111111111111112']) {
      expect(
        await r.redactResult('Runtime.evaluate', {
          result: { type: 'string', value: card },
        }),
      ).toEqual({ result: { type: 'string', value: MASK } });
    }
    // Documented over-mask: a 10-digit run in a title.
    expect(
      await r.redactResult('Runtime.evaluate', {
        result: { type: 'string', value: 'Order 1234567890 confirmed' },
      }),
    ).toEqual({ result: { type: 'string', value: MASK } });
    const thrown = JSON.stringify(
      await r.redactResult('Runtime.evaluate', {
        result: {
          type: 'object',
          subtype: 'error',
          description: 'Error: 900-00-1234',
        },
        exceptionDetails: {
          exceptionId: 1,
          text: 'Uncaught 900-00-1234',
          lineNumber: 0,
          columnNumber: 5,
          exception: { type: 'object', description: 'Error: 900001234' },
        },
      }),
    );
    expect(thrown).not.toMatch(/1234/);
    const props = JSON.stringify(
      await r.redactResult('Runtime.getProperties', {
        result: [
          { name: 'ssn', value: { type: 'string', value: '900001234' } },
          { name: 'n', value: { type: 'number', value: 3, description: '3' } },
        ],
      }),
    );
    expect(props).not.toMatch(/900001234/);
    expect(props).toContain('"description":"3"');
  });

  it('leaves booleans, null, undefined, handles, counts, coordinates, ids and box models unchanged', async () => {
    const r = createRedactor();
    const unchanged: Array<[string, unknown]> = [
      ['Runtime.evaluate', { result: { type: 'boolean', value: true } }],
      [
        'Runtime.evaluate',
        { result: { type: 'object', subtype: 'null', value: null } },
      ],
      ['Runtime.evaluate', { result: { type: 'undefined' } }],
      [
        'Runtime.callFunctionOn',
        {
          result: {
            type: 'object',
            subtype: 'node',
            className: 'HTMLInputElement',
            description: 'input#ssn',
            objectId: '-7812345678901234567.1.2',
          },
        },
      ],
      [
        'Runtime.evaluate',
        { result: { type: 'number', value: 3, description: '3' } },
      ],
      [
        'Runtime.callFunctionOn',
        {
          result: {
            type: 'object',
            value: { x: 412.3333435058594, y: 128.5, width: 300, height: 40 },
          },
        },
      ],
      [
        'DOM.getBoxModel',
        {
          model: {
            content: [
              412.3333435058594, 10, 712.3333435058594, 10, 712, 50, 412, 50,
            ],
            width: 300,
            height: 40,
          },
        },
      ],
      [
        'DOM.describeNode',
        {
          node: {
            nodeId: 0,
            backendNodeId: 123456789,
            nodeType: 1,
            nodeName: 'DIV',
            attributes: [],
          },
        },
      ],
      [
        'Target.attachToTarget',
        { sessionId: '0123ABCD1234EF5678901234ABCDEF12' },
      ],
    ];
    for (const [method, result] of unchanged) {
      expect(await r.redactResult(method, result), method).toEqual(result);
    }
  });
});

describe('review regressions, round 6: shape guard gaps (nothing registered)', () => {
  const noLeak = (x: unknown) =>
    expect(JSON.stringify(x)).not.toMatch(/900-00-1234|900001234|900 00 1234/);

  it('renames value-shaped keys of page data, uniquely, keeping __proto__ safe', async () => {
    const r = createRedactor();
    const value = JSON.parse(
      '{"900-00-1234": 1, "900 00 1234": 2, "ok": 3, "__proto__": {"x": 1}}',
    );
    const out = (await r.redactResult('Runtime.callFunctionOn', {
      result: { type: 'object', value },
    })) as { result: { value: Record<string, unknown> } };
    expect(Object.keys(out.result.value)).toEqual([
      MASK,
      `${MASK}#2`,
      'ok',
      '__proto__',
    ]);
    expect(Object.getPrototypeOf(out.result.value)).toBe(Object.prototype);
  });

  it('masks value-shaped property names in Runtime.getProperties, not protocol names', async () => {
    const r = createRedactor();
    const out = (await r.redactResult('Runtime.getProperties', {
      result: [
        { name: '900001234', value: { type: 'string', value: 'a' } },
        {
          name: 'length',
          value: { type: 'number', value: 3, description: '3' },
        },
        {
          name: '__proto__',
          value: { type: 'object', objectId: '-781234567890.1.2' },
        },
      ],
      internalProperties: [
        { name: '[[900-00-1234]]', value: { type: 'string', value: 'b' } },
      ],
    })) as {
      result: Array<{ name: string; value: Record<string, unknown> }>;
      internalProperties: Array<{ name: string }>;
    };
    expect(out.result.map((p) => p.name)).toEqual([
      MASK,
      'length',
      '__proto__',
    ]);
    expect(out.result[2].value.objectId).toBe('-781234567890.1.2');
    expect(out.internalProperties[0].name).toBe(MASK);
  });

  it('masks form bodies with empty, bare or trailing segments, encodings, and value-shaped JSON', () => {
    const r = createRedactor();
    const body = (
      postData: string,
      type = 'application/x-www-form-urlencoded',
    ) =>
      (
        r.redactEvent('Network.requestWillBeSent', {
          requestId: '1.1',
          request: {
            url: 'https://b.test/',
            method: 'POST',
            postData,
            headers: { 'Content-Type': type },
          },
        }) as { request: { postData: string } }
      ).request.postData;
    expect(body('a=1&&ssn=900-00-1234')).toBe(`a=1&&ssn=${MASK}`);
    expect(body('ssn=900-00-1234&flag')).toBe(`ssn=${MASK}&flag`);
    expect(body('x=1&v=900-00-1234&')).toBe(`x=1&v=${MASK}&`);
    expect(body('v=900%2D00%2D1234')).toBe(`v=${MASK}`);
    expect(body('v=900+00+1234')).toBe(`v=${MASK}`);
    expect(body('a=1;v=900001234')).toBe(`a=1;v=${MASK}`);
    noLeak(body('{"ssn":"900-00-1234"}'));
    noLeak(
      body(
        '--b\r\nContent-Disposition: form-data; name="x"\r\n\r\n900-00-1234\r\n--b--',
      ),
    );
    expect(body('a=1&b=two')).toBe('a=1&b=two');
  });

  it('masks URL userinfo, value-shaped path segments, ; parameters and path-like fragments', () => {
    const r = createRedactor();
    const url = (u: string) =>
      (
        r.redactEvent('Page.navigatedWithinDocument', { url: u }) as {
          url: string;
        }
      ).url;
    expect(url('https://900-00-1234@host.test/')).toBe(
      `https://${MASK}@host.test/`,
    );
    expect(url('https://user:900001234@host.test/a')).toBe(
      `https://${MASK}@host.test/a`,
    );
    expect(url('https://b.test/ssn/900-00-1234/edit')).toBe(
      `https://b.test/ssn/${MASK}/edit`,
    );
    expect(url('https://b.test/p/900%2D00%2D1234')).toBe(
      `https://b.test/p/${MASK}`,
    );
    expect(url('https://b.test/a;ssn=900-00-1234/b')).toBe(
      `https://b.test/a;ssn=${MASK}/b`,
    );
    expect(url('https://b.test/q?a=1;ssn=900-00-1234')).toBe(
      `https://b.test/q?a=1;ssn=${MASK}`,
    );
    expect(url('https://b.test/#ssn=900-00-1234')).toBe(
      `https://b.test/#ssn=${MASK}`,
    );
    expect(url('https://b.test/#/ssn/900001234')).toBe(
      `https://b.test/#${MASK}`,
    );
    expect(url('https://b.test/list/2;page=3?sort=name')).toBe(
      'https://b.test/list/2;page=3?sort=name',
    );
  });

  it('applies the shape guard to Runtime events', () => {
    const r = createRedactor();
    noLeak(
      r.redactEvent('Runtime.consoleAPICalled', {
        type: 'log',
        args: [
          { type: 'string', value: 'ssn 900-00-1234' },
          { type: 'number', value: 900001234, description: '900001234' },
          { type: 'object', value: { '900-00-1234': true } },
        ],
        executionContextId: 1,
        timestamp: 1792786214322.5,
        stackTrace: {
          callFrames: [
            {
              functionName: 'f900001234',
              scriptId: '12',
              url: 'https://b.test/a.js',
              lineNumber: 1,
              columnNumber: 2,
            },
          ],
        },
      }),
    );
    noLeak(
      r.redactEvent('Runtime.exceptionThrown', {
        timestamp: 1,
        exceptionDetails: {
          exceptionId: 1,
          text: 'Uncaught 900-00-1234',
          lineNumber: 0,
          columnNumber: 0,
          exception: {
            type: 'object',
            subtype: 'error',
            description: 'Error: 900001234',
          },
        },
      }),
    );
    noLeak(
      r.redactEvent('Runtime.inspectRequested', {
        object: { type: 'string', value: '900-00-1234' },
        hints: {},
      }),
    );
    noLeak(
      r.redactEvent('Runtime.bindingCalled', {
        name: 'cb',
        payload: '{"ssn":"900-00-1234"}',
        executionContextId: 1,
      }),
    );
  });

  it('applies the shape guard to dialog, frame, target, network and DOM page text', async () => {
    const r = createRedactor();
    noLeak(
      r.redactEvent('Page.javascriptDialogOpening', {
        url: 'https://b.test/',
        message: 'Confirm SSN 900-00-1234?',
        type: 'prompt',
        defaultPrompt: '900001234',
      }),
    );
    noLeak(
      r.redactEvent('Page.frameNavigated', {
        frame: {
          id: 'ABCDEF0123456789AB1234CDEF012345',
          name: 'f-900001234',
          url: 'https://b.test/',
        },
      }),
    );
    noLeak(
      r.redactEvent('Target.targetInfoChanged', {
        targetInfo: {
          targetId: '0123ABCD1234EF5678901234ABCDEF12',
          title: 'Case 900-00-1234',
          url: 'https://b.test/',
        },
      }),
    );
    noLeak(
      r.redactEvent('Network.responseReceived', {
        requestId: '1.1',
        response: {
          url: 'https://b.test/',
          status: 200,
          statusText: 'OK',
          headers: { 'X-Ssn': '900-00-1234' },
        },
      }),
    );
    noLeak(
      r.redactEvent('Network.webSocketFrameReceived', {
        requestId: '1.1',
        response: { opcode: 1, mask: false, payloadData: '{"v":"900001234"}' },
      }),
    );
    noLeak(
      await r.redactResult('DOM.describeNode', {
        node: {
          nodeType: 1,
          nodeName: 'DIV',
          attributes: ['data-x', '900-00-1234'],
        },
      }),
    );
    // Ids, enums and timestamps survive.
    const lifecycle = {
      frameId: 'ABCDEF0123456789AB1234CDEF012345',
      loaderId: 'ABCDEF0123456789AB1234CDEF012345',
      name: 'load',
      timestamp: 1234567.89,
    };
    expect(r.redactEvent('Page.lifecycleEvent', lifecycle)).toEqual(lifecycle);
    const box = {
      model: {
        content: [412.3333435058594, 10, 712, 10, 712, 50, 412, 50],
        width: 300,
        height: 40,
      },
    };
    expect(await r.redactResult('DOM.getBoxModel', box)).toEqual(box);
  });
});

describe('review regressions, round 7: request body handling by content type', () => {
  const body = (postData: string, headers?: Record<string, string>) => {
    const r = createRedactor();
    return (
      r.redactEvent('Network.requestWillBeSent', {
        requestId: '1.1',
        request: {
          url: 'https://b.test/',
          method: 'POST',
          postData,
          ...(headers ? { headers } : {}),
        },
      }) as { request: { postData: string } }
    ).request.postData;
  };
  const form = { 'Content-Type': 'application/x-www-form-urlencoded' };

  it('a body mentioning content-disposition is still masked by parameter name', () => {
    expect(body('x=content-disposition&password=hunter2', form)).toBe(
      `x=content-disposition&password=${MASK}`,
    );
    // No content type: ambiguous, masked whole.
    expect(body('x=content-disposition&password=hunter2')).toBe(MASK);
  });

  it('a urlencoded body with a -- line is masked by parameter name', () => {
    expect(body('a=1&password=hunter2\n--x', form)).not.toContain('hunter2');
  });

  it('reads Content-Type case-insensitively, with parameters', () => {
    for (const key of ['content-type', 'CONTENT-TYPE', 'Content-type']) {
      expect(
        body('password=hunter2&a=1', {
          [key]: 'Application/X-WWW-Form-Urlencoded; charset=UTF-8',
        }),
      ).toBe(`password=${MASK}&a=1`);
    }
    expect(body('pin=1234', { 'content-type': 'text/plain' })).toBe(
      `pin=${MASK}`,
    );
  });

  it('masks JSON leaves by key name and by shape', () => {
    const out = body(
      '{"password":"hunter2","user":{"pin":1234,"note":"900-00-1234"},"ok":"yes"}',
      { 'Content-Type': 'application/json' },
    );
    expect(JSON.parse(out)).toEqual({
      password: MASK,
      user: { pin: MASK, note: MASK },
      ok: 'yes',
    });
    // No content type, but it parses as JSON.
    expect(JSON.parse(body('{"password":"hunter2"}'))).toEqual({
      password: MASK,
    });
    // Declared JSON that does not parse: ambiguous, masked whole.
    expect(
      body('{"password":"hunter2"', { 'Content-Type': 'application/json' }),
    ).toBe(MASK);
    // Ordinary JSON is returned as sent.
    const plain = '{ "a": 1, "b": "two" }';
    expect(body(plain, { 'Content-Type': 'application/json' })).toBe(plain);
  });

  it('masks multipart parts by Content-Disposition name and by shape', () => {
    const multipart = [
      '--XyZ',
      'Content-Disposition: form-data; name="password"',
      '',
      'hunter2',
      '--XyZ',
      'Content-Disposition: form-data; name="note"',
      '',
      'case 900-00-1234',
      '--XyZ',
      'Content-Disposition: form-data; name="first"',
      '',
      'Ada',
      '--XyZ--',
      '',
    ].join('\r\n');
    const out = body(multipart, {
      'Content-Type': 'multipart/form-data; boundary=XyZ',
    });
    expect(out).not.toContain('hunter2');
    expect(out).not.toContain('900-00-1234');
    expect(out).toContain('Ada');
    expect(out).toContain('name="password"');
    // Without a boundary in the content type: ambiguous, masked whole.
    expect(body(multipart, { 'Content-Type': 'multipart/form-data' })).toBe(
      MASK,
    );
  });

  it('masks an XML body with a sensitive tag or a value shape whole', () => {
    const xml = { 'Content-Type': 'application/xml' };
    expect(body('<user><password>hunter2</password></user>', xml)).toBe(MASK);
    expect(body('<n>900-00-1234</n>', xml)).toBe(MASK);
    expect(body('<user><first>Ada</first></user>', xml)).toBe(
      '<user><first>Ada</first></user>',
    );
    // No content type: the tag name still masks it.
    expect(body('<password>hunter2</password>')).toBe(MASK);
  });

  it('masks an ambiguous body whole when a name or shape is found', () => {
    expect(body('password=hunter2\n--x')).toBe(MASK);
    expect(body('note: 900 00 1234')).toBe(MASK);
    expect(
      body('--b\r\nContent-Disposition: form-data; name="pin"\r\n\r\n1234'),
    ).toBe(MASK);
  });

  it('leaves an ordinary urlencoded body unchanged', () => {
    expect(body('first=Ada&last=Lovelace&page=2', form)).toBe(
      'first=Ada&last=Lovelace&page=2',
    );
    expect(body('first=Ada&last=Lovelace')).toBe('first=Ada&last=Lovelace');
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
