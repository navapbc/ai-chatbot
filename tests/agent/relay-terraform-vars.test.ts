import { spawnSync } from 'node:child_process';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

// scripts/relay-terraform-vars.sh turns the relay GitHub environment variables
// into TF_VAR_ lines for $GITHUB_ENV in deploy.yml and promote.yml. Unset
// means no line (Terraform default); an invalid value fails with a fixed
// message that never carries the value, and prints nothing to stdout.

const script = join(process.cwd(), 'scripts/relay-terraform-vars.sh');

const run = (vars: Record<string, string>) => {
  const result = spawnSync('bash', [script], {
    env: { PATH: process.env.PATH ?? '/usr/bin:/bin', ...vars },
    encoding: 'utf8',
  });
  return {
    status: result.status,
    stdout: result.stdout,
    stderr: result.stderr,
  };
};

const ORIGIN = `chrome-extension://${'abcdefghijklmnop'.repeat(2)}`;
const invalidMessage = (name: string) =>
  `relay-terraform-vars: ${name} is not valid (see docs/rollout-pilot.md section 4.1)\n`;

describe('relay-terraform-vars.sh', () => {
  it('prints nothing when no variable is set', () => {
    expect(run({})).toEqual({ status: 0, stdout: '', stderr: '' });
  });

  it.each([
    ['RELAY_RUNS_ENABLED', 'true', 'TF_VAR_relay_runs_enabled=true'],
    ['RELAY_RUNS_ENABLED', 'false', 'TF_VAR_relay_runs_enabled=false'],
    [
      'RELAY_TOKEN_ENDPOINT_ENABLED',
      'true',
      'TF_VAR_relay_token_endpoint_enabled=true',
    ],
    [
      'RELAY_COMMAND_DURATION_METRIC_ENABLED',
      'true',
      'TF_VAR_relay_command_duration_metric_enabled=true',
    ],
    ['RELAY_MIN_INSTANCES', '1', 'TF_VAR_relay_min_instances=1'],
    ['RELAY_MIN_INSTANCES', '0', 'TF_VAR_relay_min_instances=0'],
    [
      'RELAY_EXTENSION_ORIGIN',
      ORIGIN,
      `TF_VAR_relay_extension_origin=${ORIGIN}`,
    ],
    [
      'RELAY_ALLOWED_MODELS',
      'claude-opus-4-7, gpt-5/mini:1',
      'TF_VAR_relay_allowed_models=claude-opus-4-7,gpt-5/mini:1',
    ],
    [
      'RELAY_ALLOWED_HOSTS',
      'labs-asp-relay-dev-123.us-central1.run.app',
      'TF_VAR_relay_allowed_hosts=labs-asp-relay-dev-123.us-central1.run.app',
    ],
    [
      'RELAY_ALLOWED_HOSTS',
      ' A.Run.App , b.example ',
      'TF_VAR_relay_allowed_hosts=a.run.app,b.example',
    ],
    [
      'RELAY_ALLOWED_HOST_SUFFIXES',
      '.run.app',
      'TF_VAR_relay_allowed_host_suffixes=.run.app',
    ],
    [
      'RELAY_ALLOWED_HOST_SUFFIXES',
      '.Run.App,example.org',
      'TF_VAR_relay_allowed_host_suffixes=.run.app,example.org',
    ],
  ])('%s=%j prints %s', (name, value, line) => {
    expect(run({ [name]: value })).toEqual({
      status: 0,
      stdout: `${line}\n`,
      stderr: '',
    });
  });

  it('prints every set variable together', () => {
    const result = run({
      RELAY_RUNS_ENABLED: 'true',
      RELAY_MIN_INSTANCES: '1',
      RELAY_ALLOWED_HOSTS: 'relay.example',
    });
    expect(result.status).toBe(0);
    expect(result.stdout).toBe(
      [
        'TF_VAR_relay_runs_enabled=true',
        'TF_VAR_relay_min_instances=1',
        'TF_VAR_relay_allowed_hosts=relay.example',
        '',
      ].join('\n'),
    );
  });

  it.each([
    'RELAY_RUNS_ENABLED',
    'RELAY_TOKEN_ENDPOINT_ENABLED',
    'RELAY_EXTENSION_ORIGIN',
    'RELAY_ALLOWED_MODELS',
    'RELAY_ALLOWED_HOSTS',
    'RELAY_ALLOWED_HOST_SUFFIXES',
    'RELAY_MIN_INSTANCES',
    'RELAY_COMMAND_DURATION_METRIC_ENABLED',
  ])('%s empty or blank prints nothing', (name) => {
    for (const value of ['', ' ', '   ']) {
      expect(run({ [name]: value })).toEqual({
        status: 0,
        stdout: '',
        stderr: '',
      });
    }
  });

  describe('the none sentinel for the suffix list', () => {
    it.each(['none', 'None', 'NONE', ' none '])(
      '%j prints an empty suffix list when the host list is set',
      (sentinel) => {
        expect(
          run({
            RELAY_ALLOWED_HOSTS: 'relay.example',
            RELAY_ALLOWED_HOST_SUFFIXES: sentinel,
          }),
        ).toEqual({
          status: 0,
          stdout:
            'TF_VAR_relay_allowed_hosts=relay.example\nTF_VAR_relay_allowed_host_suffixes=\n',
          stderr: '',
        });
      },
    );

    it.each(['none', 'None', 'NONE'])(
      'refuses %j without a host list (never a literal suffix)',
      (sentinel) => {
        for (const hosts of [undefined, '', ' ']) {
          const vars: Record<string, string> = {
            RELAY_ALLOWED_HOST_SUFFIXES: sentinel,
          };
          if (hosts !== undefined) vars.RELAY_ALLOWED_HOSTS = hosts;
          expect(run(vars)).toEqual({
            status: 1,
            stdout: '',
            stderr:
              'relay-terraform-vars: RELAY_ALLOWED_HOST_SUFFIXES=none needs RELAY_ALLOWED_HOSTS\n',
          });
        }
      },
    );

    it('refuses none when the host list is invalid', () => {
      expect(
        run({
          RELAY_ALLOWED_HOSTS: 'relay.example:443',
          RELAY_ALLOWED_HOST_SUFFIXES: 'none',
        }),
      ).toEqual({
        status: 1,
        stdout: '',
        stderr: invalidMessage('RELAY_ALLOWED_HOSTS'),
      });
    });
  });

  it.each([
    ['RELAY_RUNS_ENABLED', 'yes'],
    ['RELAY_RUNS_ENABLED', '1'],
    ['RELAY_RUNS_ENABLED', 'True'],
    ['RELAY_TOKEN_ENDPOINT_ENABLED', 'on'],
    ['RELAY_COMMAND_DURATION_METRIC_ENABLED', 'truex'],
    ['RELAY_MIN_INSTANCES', '2'],
    ['RELAY_MIN_INSTANCES', '0.5'],
    ['RELAY_MIN_INSTANCES', '01'],
    ['RELAY_EXTENSION_ORIGIN', `chrome-extension://${'q'.repeat(32)}`],
    ['RELAY_EXTENSION_ORIGIN', `chrome-extension://${'a'.repeat(31)}`],
    ['RELAY_EXTENSION_ORIGIN', `chrome-extension://${'a'.repeat(33)}`],
    ['RELAY_EXTENSION_ORIGIN', `https://${'a'.repeat(32)}`],
    ['RELAY_EXTENSION_ORIGIN', `${ORIGIN}/`],
    ['RELAY_ALLOWED_HOSTS', 'relay.example:443'],
    ['RELAY_ALLOWED_HOSTS', 'relay.example/path'],
    ['RELAY_ALLOWED_HOSTS', 'https://relay.example'],
    ['RELAY_ALLOWED_HOSTS', 'relay example'],
    ['RELAY_ALLOWED_HOSTS', 'relay.example,'],
    ['RELAY_ALLOWED_HOSTS', 'a,,b'],
    ['RELAY_ALLOWED_HOSTS', '.run.app'],
    ['RELAY_ALLOWED_HOSTS', 'relay.example-'],
    ['RELAY_ALLOWED_HOSTS', 'relay_host'],
    ['RELAY_ALLOWED_HOST_SUFFIXES', '.run.app,'],
    ['RELAY_ALLOWED_HOST_SUFFIXES', '..run.app'],
    ['RELAY_ALLOWED_HOST_SUFFIXES', '.run.app:443'],
    ['RELAY_ALLOWED_HOST_SUFFIXES', 'none,.run.app'],
    ['RELAY_ALLOWED_HOST_SUFFIXES', '.run.app, None'],
    ['RELAY_ALLOWED_HOSTS', 'none'],
    ['RELAY_ALLOWED_HOSTS', 'relay.example,NONE'],
    ['RELAY_ALLOWED_MODELS', 'claude opus'],
    ['RELAY_ALLOWED_MODELS', 'a,,b'],
    ['RELAY_ALLOWED_MODELS', 'a,'],
    ['RELAY_ALLOWED_MODELS', 'a=b'],
    ['RELAY_ALLOWED_MODELS', 'a;b'],
    ['RELAY_ALLOWED_MODELS', 'a\nTF_VAR_relay_runs_enabled=true'],
    ['RELAY_ALLOWED_HOSTS', 'a.example\nTF_VAR_relay_runs_enabled=true'],
    ['RELAY_EXTENSION_ORIGIN', `${ORIGIN}\nTF_VAR_relay_runs_enabled=true`],
    ['RELAY_RUNS_ENABLED', 'true\nTF_VAR_relay_min_instances=1'],
    ['RELAY_MIN_INSTANCES', '1\r'],
    ['RELAY_ALLOWED_MODELS', 'a\tb'],
    ['RELAY_ALLOWED_HOSTS', 'a\u0007b'],
  ])('%s=%j fails with a fixed message', (name, value) => {
    const result = run({ [name]: value });
    expect(result).toEqual({
      status: 1,
      stdout: '',
      stderr: invalidMessage(name),
    });
  });

  it('prints nothing on failure even when earlier variables were valid', () => {
    const result = run({
      RELAY_RUNS_ENABLED: 'true',
      RELAY_MIN_INSTANCES: '1',
      RELAY_ALLOWED_HOST_SUFFIXES: 'secret-ish value',
    });
    expect(result.status).toBe(1);
    expect(result.stdout).toBe('');
    expect(result.stderr).toBe(invalidMessage('RELAY_ALLOWED_HOST_SUFFIXES'));
    expect(result.stderr).not.toContain('secret-ish');
  });
});
