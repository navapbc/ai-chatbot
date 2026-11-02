import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

// scripts/guard-deploy-ref.sh is the first step of deploy.yml (setup),
// promote.yml (promote) and cleanup-preview.yml (cleanup). It refuses the
// extension branch on a dispatch or pull request, dev from any branch but
// develop, and prod from any branch but main, with one fixed line that never
// carries an input value.

const root = process.cwd();
const script = join(root, 'scripts/guard-deploy-ref.sh');
const read = (p: string) => readFileSync(join(root, p), 'utf8');

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

const refusal = (reason: string) =>
  `::error::guard-deploy-ref: ${reason} (see docs/rollout-pilot.md section 4.7)\n`;
const EXTENSION = refusal(
  'the extension branch is never deployed to the shared environments',
);
const DEV = refusal('dev can only be dispatched from develop');
const PROD = refusal('prod can only be dispatched from main');

const dispatch = (ref: string, environment: string) =>
  run({
    GITHUB_EVENT_NAME: 'workflow_dispatch',
    GUARD_REF_NAME: ref,
    GUARD_HEAD_REF: '',
    GUARD_ENVIRONMENT: environment,
  });

const pass = { status: 0, stdout: '', stderr: '' };
const fail = (stderr: string) => ({ status: 1, stdout: '', stderr });

describe('guard-deploy-ref.sh', () => {
  it.each(['dev', 'preview', 'prod'])(
    'refuses a dispatch of %s from browser-extension',
    (environment) => {
      expect(dispatch('browser-extension', environment)).toEqual(
        fail(EXTENSION),
      );
    },
  );

  it.each([
    ['dev', 'develop', pass],
    ['dev', 'feature/x', fail(DEV)],
    ['dev', 'main', fail(DEV)],
    ['prod', 'main', pass],
    ['prod', 'develop', fail(PROD)],
    ['prod', 'feature/x', fail(PROD)],
    ['preview', 'feature/x', pass],
    ['preview', 'develop', pass],
  ])('dispatch of %s from %s', (environment, ref, expected) => {
    expect(dispatch(ref, environment)).toEqual(expected);
  });

  it('allows a dispatch with no environment from another branch', () => {
    expect(dispatch('feature/x', '')).toEqual(pass);
  });

  it('refuses a pull request from browser-extension', () => {
    expect(
      run({
        GITHUB_EVENT_NAME: 'pull_request',
        GUARD_REF_NAME: '12/merge',
        GUARD_HEAD_REF: 'browser-extension',
        GUARD_ENVIRONMENT: '',
      }),
    ).toEqual(fail(EXTENSION));
  });

  it('allows a pull request from another branch', () => {
    expect(
      run({
        GITHUB_EVENT_NAME: 'pull_request',
        GUARD_REF_NAME: '12/merge',
        GUARD_HEAD_REF: 'feature/x',
        GUARD_ENVIRONMENT: '',
      }),
    ).toEqual(pass);
  });

  it('leaves push events alone', () => {
    expect(
      run({
        GITHUB_EVENT_NAME: 'push',
        GUARD_REF_NAME: 'browser-extension',
        GUARD_HEAD_REF: '',
        GUARD_ENVIRONMENT: '',
      }),
    ).toEqual(pass);
  });

  it('passes an unknown event through', () => {
    expect(
      run({
        GITHUB_EVENT_NAME: 'repository_dispatch',
        GUARD_REF_NAME: 'browser-extension',
        GUARD_HEAD_REF: 'browser-extension',
        GUARD_ENVIRONMENT: 'prod',
      }),
    ).toEqual(pass);
  });

  it('treats missing variables as empty without crashing', () => {
    expect(run({})).toEqual(pass);
    expect(run({ GITHUB_EVENT_NAME: 'workflow_dispatch' })).toEqual(pass);
    expect(run({ GITHUB_EVENT_NAME: 'pull_request' })).toEqual(pass);
    expect(
      run({ GITHUB_EVENT_NAME: 'workflow_dispatch', GUARD_ENVIRONMENT: 'dev' }),
    ).toEqual(fail(DEV));
  });

  it('never echoes an input value', () => {
    const marker = 'feature/MARKER-7f3a';
    const results = [
      dispatch(marker, 'dev'),
      dispatch(marker, 'prod'),
      run({
        GITHUB_EVENT_NAME: 'workflow_dispatch',
        GUARD_REF_NAME: 'browser-extension',
        GUARD_ENVIRONMENT: 'MARKER-env-9c1d',
      }),
      run({
        GITHUB_EVENT_NAME: 'pull_request',
        GUARD_REF_NAME: 'MARKER-ref-2b8e',
        GUARD_HEAD_REF: 'browser-extension',
      }),
    ];
    for (const result of results) {
      expect(result.status).toBe(1);
      expect(result.stderr).not.toContain('MARKER');
      expect(result.stdout).toBe('');
    }
  });
});

/** The `- name:` / `- uses:` step blocks of a job, in order. */
const jobSteps = (workflow: string, job: string): string[] => {
  const start = workflow.indexOf(`\n  ${job}:\n`);
  expect(start, `job ${job} exists`).toBeGreaterThan(-1);
  const rest = workflow.slice(start + job.length + 4);
  const next = rest.search(/\n {2}[a-z][a-z0-9_-]*:\n/);
  const body = next === -1 ? rest : rest.slice(0, next);
  const stepsAt = body.indexOf('\n    steps:\n');
  expect(stepsAt, `job ${job} has steps`).toBeGreaterThan(-1);
  return body
    .slice(stepsAt)
    .split(/\n {6}- /)
    .slice(1)
    .map((step) =>
      step
        .split('\n')
        .filter((line) => !line.trim().startsWith('#'))
        .join('\n'),
    );
};

const GUARD_ENV = [
  'GITHUB_EVENT_NAME: ${{ github.event_name }}',
  'GUARD_REF_NAME: ${{ github.ref_name }}',
  'GUARD_HEAD_REF: ${{ github.head_ref }}',
];

const expectGuardStep = (step: string, scriptPath: string) => {
  expect(step).toContain('name: Guard deploy ref');
  for (const line of GUARD_ENV) expect(step).toContain(line);
  expect(step).toContain(`run: bash ${scriptPath}\n`);
  // The run line takes no expression: values come through env only.
  const runLine = step.split('\n').find((l) => l.trim().startsWith('run:'));
  expect(runLine).not.toContain('${{');
};

describe('deploy.yml guard', () => {
  const steps = jobSteps(read('.github/workflows/deploy.yml'), 'setup');

  it('checks out only the guard script, then runs it first', () => {
    expect(steps[0]).toContain('uses: actions/checkout@v4');
    expect(steps[0]).toContain('persist-credentials: false');
    expect(steps[0]).toContain('sparse-checkout: scripts/guard-deploy-ref.sh');
    expect(steps[0]).toContain('sparse-checkout-cone-mode: false');
    expectGuardStep(steps[1], 'scripts/guard-deploy-ref.sh');
    expect(steps[1]).toContain(
      'GUARD_ENVIRONMENT: ${{ github.event.inputs.environment }}',
    );
    expect(steps[2]).toContain('name: Determine environment');
  });
});

describe('promote.yml guard', () => {
  const steps = jobSteps(read('.github/workflows/promote.yml'), 'promote');

  it('runs the guard from github.sha before the tag checkout', () => {
    expect(steps[0]).toContain('uses: actions/checkout@v4');
    expect(steps[0]).toContain('ref: ${{ github.sha }}');
    expect(steps[0]).toContain('path: guard-ref');
    expect(steps[0]).toContain('persist-credentials: false');
    expect(steps[0]).toContain('sparse-checkout: scripts/guard-deploy-ref.sh');
    expect(steps[0]).toContain('sparse-checkout-cone-mode: false');
    expectGuardStep(steps[1], 'guard-ref/scripts/guard-deploy-ref.sh');
    expect(steps[1]).toContain(
      'GUARD_ENVIRONMENT: ${{ github.event.inputs.environment }}',
    );
    expect(steps[2]).toContain('name: Checkout code at specified version');
    expect(steps[2]).toContain('ref: ${{ github.event.inputs.version }}');
  });
});

describe('cleanup-preview.yml guard', () => {
  const steps = jobSteps(
    read('.github/workflows/cleanup-preview.yml'),
    'cleanup',
  );

  it('runs the guard right after checkout, before any cloud step', () => {
    expect(steps[0]).toContain('uses: actions/checkout@v4');
    expectGuardStep(steps[1], 'scripts/guard-deploy-ref.sh');
    expect(steps[1]).toContain("GUARD_ENVIRONMENT: ''");
    expect(steps[2]).toContain('name: Authenticate to Google Cloud');
  });
});
