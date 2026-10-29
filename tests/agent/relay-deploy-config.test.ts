import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

// Pins the pipeline and Terraform shape of the relay rollout: every relay
// setting defaults to today's behavior (relay off, flags unset), promotion
// fails without a relay image only when the relay is enabled, and preview
// environments never get a relay.

const root = process.cwd();
const read = (p: string) => readFileSync(join(root, p), 'utf8');

/** Top-level `variable "<name>" { ... }` blocks, keyed by name. */
const variableBlocks = (hcl: string): Map<string, string> => {
  const blocks = new Map<string, string>();
  const parts = hcl.split(/^variable "/m).slice(1);
  for (const part of parts) {
    const name = part.slice(0, part.indexOf('"'));
    blocks.set(name, part);
  }
  return blocks;
};

/** The `default = ...` value of a block, ignoring comment lines. */
const defaultOf = (block: string): string | undefined => {
  for (const line of block.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('#')) continue;
    const match = /^default\s*=\s*(.+)$/.exec(trimmed);
    if (match) return match[1].trim();
  }
  return undefined;
};

describe('terraform/variables.tf relay defaults', () => {
  const blocks = variableBlocks(read('terraform/variables.tf'));

  it.each([
    ['relay_image_url', '""'],
    ['relay_max_instances', '1'],
    ['relay_runs_enabled', 'false'],
    ['relay_token_endpoint_enabled', 'false'],
    ['relay_extension_origin', '""'],
    ['relay_allowed_models', '""'],
    ['relay_allowed_hosts', '""'],
    ['relay_allowed_host_suffixes', '".run.app"'],
    ['relay_min_instances', '0'],
    ['relay_command_duration_metric_enabled', 'false'],
  ])('%s defaults to %s', (name, expected) => {
    const block = blocks.get(name);
    expect(block, `variable ${name} is declared`).toBeDefined();
    expect(defaultOf(block ?? '')).toBe(expected);
  });

  it('validates the extension origin shape', () => {
    expect(blocks.get('relay_extension_origin')).toContain(
      '^chrome-extension://[a-p]{32}$',
    );
  });

  it('allows only 0 or 1 minimum relay instances', () => {
    expect(blocks.get('relay_min_instances')).toContain(
      'condition     = contains([0, 1], var.relay_min_instances)',
    );
  });

  it('accepts only bare host names in relay_allowed_hosts', () => {
    const block = blocks.get('relay_allowed_hosts') ?? '';
    expect(block).toContain('trimspace(host) != ""');
    expect(block).toContain(
      'can(regex("^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$", lower(trimspace(host))))',
    );
  });

  it('refuses an empty entry in relay_allowed_host_suffixes', () => {
    expect(blocks.get('relay_allowed_host_suffixes')).toContain(
      'trimspace(suffix) != ""',
    );
  });
});

describe('terraform relay wiring', () => {
  const relay = read('terraform/relay.tf');
  const cloudRun = read('terraform/cloud_run.tf');

  it('reads eve-internal-secret at plan time only when the relay is enabled', () => {
    expect(relay).toMatch(
      /data "google_secret_manager_secret" "eve_internal" \{\s*count\s*=\s*local\.relay_enabled \? 1 : 0/,
    );
    expect(relay).toContain('secret_id = local.eve_internal_secret_name');
  });

  it('sets the run and token flags only when their variable is true', () => {
    for (const [variable, envName] of [
      ['relay_runs_enabled', 'RELAY_RUNS_ENABLED'],
      ['relay_token_endpoint_enabled', 'RELAY_TOKEN_ENDPOINT_ENABLED'],
    ]) {
      expect(cloudRun).toMatch(
        new RegExp(
          `for_each = local\\.relay_enabled && var\\.${variable} \\? \\[1\\] : \\[\\]\\s*content \\{\\s*name\\s*=\\s*"${envName}"\\s*value\\s*=\\s*"1"`,
        ),
      );
    }
  });

  it('sets the origin and model allowlist only with the relay and a value', () => {
    for (const [variable, envName] of [
      ['relay_extension_origin', 'RELAY_EXTENSION_ORIGIN'],
      ['relay_allowed_models', 'RELAY_ALLOWED_MODELS'],
    ]) {
      expect(cloudRun).toMatch(
        new RegExp(
          `for_each = local\\.relay_enabled && var\\.${variable} != "" \\? \\[1\\] : \\[\\]\\s*content \\{\\s*name\\s*=\\s*"${envName}"\\s*value\\s*=\\s*var\\.${variable}`,
        ),
      );
    }
  });

  it('sets the relay host lists only when their variable is non-empty', () => {
    for (const [variable, envName] of [
      ['relay_allowed_host_suffixes', 'RELAY_ALLOWED_HOST_SUFFIXES'],
      ['relay_allowed_hosts', 'RELAY_ALLOWED_HOSTS'],
    ]) {
      expect(relay).toMatch(
        new RegExp(
          `for_each = var\\.${variable} != "" \\? \\[1\\] : \\[\\]\\s*content \\{\\s*name\\s*=\\s*"${envName}"\\s*value\\s*=\\s*var\\.${variable}`,
        ),
      );
    }
    expect(relay).not.toMatch(/value\s*=\s*"\.run\.app"/);
  });

  it('refuses a relay with no host list at all', () => {
    expect(relay).toContain(
      'condition     = var.relay_allowed_hosts != "" || var.relay_allowed_host_suffixes != ""',
    );
  });

  it('ties the extension origin to extension/EXTENSION_ID', () => {
    expect(relay).toContain(
      'condition     = var.relay_extension_origin == "" || var.relay_extension_origin == "chrome-extension://${trimspace(file("${path.module}/../extension/EXTENSION_ID"))}"',
    );
    expect(relay).toMatch(
      /P5-g[^\n]*RELAY_EXTENSION_ID[^\n]*relay_extension_origin/,
    );
  });

  it('no longer says the token route waits for plan 4', () => {
    expect(cloudRun).not.toContain('until plan 4 turns it on');
  });

  it('keeps the relay at one instance and the minimum on a variable', () => {
    expect(relay).toContain('min_instance_count = var.relay_min_instances');
    expect(relay).toContain('max_instance_count = var.relay_max_instances');
  });

  it('gates the command metric on its variable and matches only command lines', () => {
    expect(relay).toMatch(
      /resource "google_logging_metric" "relay_command_duration" \{\s*count\s*=\s*local\.relay_enabled && var\.relay_command_duration_metric_enabled \? 1 : 0/,
    );
    expect(relay).toContain('jsonPayload.event=\\"command\\"');
    expect(relay).toContain('EXTRACT(jsonPayload.durationMs)');
    expect(relay).not.toContain('label_extractors');
  });
});

describe('promote.yml relay image check', () => {
  const promote = read('.github/workflows/promote.yml');
  const verify = promote.slice(
    promote.indexOf('- name: Verify images exist'),
    promote.indexOf('- name: Setup Terraform'),
  );
  // Split the step at the `if` that tests the relay flag and at the `else`
  // and `fi` lines with that same indentation, so nested ifs and words that
  // contain "else" or "fi" do not move the branch boundaries.
  const lines = verify.split('\n');
  const ifLine = lines.findIndex(
    (line) =>
      line.trim() === 'if [[ "${{ vars.RELAY_ENABLED }}" == "true" ]]; then',
  );
  const indent = ifLine === -1 ? '' : lines[ifLine].replace(/\S.*$/, '');
  const atIndent = (keyword: string) =>
    lines.findIndex((line, i) => i > ifLine && line === `${indent}${keyword}`);
  const elseLine = atIndent('else');
  const fiLine = atIndent('fi');

  it('fails the promotion when the relay is enabled and its image is missing', () => {
    expect(ifLine).toBeGreaterThan(-1);
    expect(elseLine).toBeGreaterThan(ifLine);
    const enabled = lines.slice(ifLine + 1, elseLine).join('\n');
    expect(enabled).toContain('relay_image');
    expect(enabled).toContain('::error::');
    expect(enabled).toContain('exit 1');
    expect(enabled).not.toContain('|| echo');
  });

  it('only notes the relay when it is not enabled', () => {
    expect(fiLine).toBeGreaterThan(elseLine);
    const disabled = lines.slice(elseLine + 1, fiLine).join('\n');
    expect(disabled).toContain('echo');
    expect(disabled).not.toContain('exit 1');
  });

  it('keeps the chatbot image check warn-only', () => {
    expect(verify).toContain('|| echo "Warning: chatbot image not found"');
  });
});

describe('deploy.yml relay scope', () => {
  const deploy = read('.github/workflows/deploy.yml');

  it('decides the relay once, for dev and prod only', () => {
    const uses = deploy.match(/vars\.RELAY_ENABLED/g) ?? [];
    expect(uses).toHaveLength(1);
    const setup = deploy.slice(
      deploy.indexOf('  setup:'),
      deploy.indexOf('  build:'),
    );
    expect(setup).toContain(
      'relay_enabled: ${{ steps.env.outputs.relay_enabled }}',
    );
    expect(setup).toContain('vars.RELAY_ENABLED');
    expect(setup).toMatch(
      /\[\[ "\$ENVIRONMENT" == "dev" \]\] \|\| \[\[ "\$ENVIRONMENT" == "prod" \]\]/,
    );
  });

  it('builds and passes the relay image only when setup enabled it', () => {
    expect(deploy).toContain(
      "relay_image: ${{ needs.setup.outputs.relay_enabled == 'true' && steps.images.outputs.relay_image || '' }}",
    );
    expect(deploy).toMatch(
      /- name: Build and push relay image\s*\n\s*if: needs\.setup\.outputs\.relay_enabled == 'true'/,
    );
  });
});
