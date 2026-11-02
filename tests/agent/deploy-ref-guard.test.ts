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
const NOT_BRANCH = refusal('a deploy can only be dispatched from a branch');
const ENVIRONMENT = refusal('the environment must be dev, preview or prod');
const DEV = refusal('dev can only be dispatched from develop');
const PROD = refusal('prod can only be dispatched from main');

const dispatch = (ref: string, environment: string) =>
  run({
    GUARD_EVENT_NAME: 'workflow_dispatch',
    GUARD_REF: ref,
    GUARD_HEAD_REF: '',
    GUARD_ENVIRONMENT: environment,
  });

const pull = (headRef: string) =>
  run({
    GUARD_EVENT_NAME: 'pull_request',
    GUARD_REF: 'refs/pull/12/merge',
    GUARD_HEAD_REF: headRef,
    GUARD_ENVIRONMENT: '',
  });

const pass = { status: 0, stdout: '', stderr: '' };
const fail = (stderr: string) => ({ status: 1, stdout: '', stderr });

describe('guard-deploy-ref.sh', () => {
  it.each(['dev', 'preview', 'prod'])(
    'refuses a dispatch of %s from browser-extension',
    (environment) => {
      expect(dispatch('refs/heads/browser-extension', environment)).toEqual(
        fail(EXTENSION),
      );
    },
  );

  it.each([
    ['dev', 'refs/heads/develop', pass],
    ['dev', 'refs/heads/feature/x', fail(DEV)],
    ['dev', 'refs/heads/main', fail(DEV)],
    ['prod', 'refs/heads/main', pass],
    ['prod', 'refs/heads/develop', fail(PROD)],
    ['prod', 'refs/heads/feature/x', fail(PROD)],
    ['preview', 'refs/heads/feature/x', pass],
    ['preview', 'refs/heads/develop', pass],
  ])('dispatch of %s from %s', (environment, ref, expected) => {
    expect(dispatch(ref, environment)).toEqual(expected);
  });

  it.each([
    ['dev', 'refs/tags/develop'],
    ['prod', 'refs/tags/main'],
    ['preview', 'refs/tags/browser-extension'],
    ['preview', 'refs/tags/v1.2.3'],
    ['dev', 'develop'],
    ['prod', 'main'],
    ['preview', 'browser-extension'],
    ['preview', ''],
  ])('refuses a dispatch of %s from %s (not a branch ref)', (env, ref) => {
    expect(dispatch(ref, env)).toEqual(fail(NOT_BRANCH));
  });

  it.each(['', 'DEV', 'Prod', 'dev\n', ' dev', 'preview-pr-1', 'staging'])(
    'refuses a dispatch with environment %j',
    (environment) => {
      expect(dispatch('refs/heads/develop', environment)).toEqual(
        fail(ENVIRONMENT),
      );
    },
  );

  it('refuses a pull request from browser-extension', () => {
    expect(pull('browser-extension')).toEqual(fail(EXTENSION));
  });

  it('allows a pull request from another branch', () => {
    expect(pull('feature/x')).toEqual(pass);
  });

  it('leaves push events alone', () => {
    expect(
      run({
        GUARD_EVENT_NAME: 'push',
        GUARD_REF: 'refs/heads/browser-extension',
        GUARD_HEAD_REF: '',
        GUARD_ENVIRONMENT: '',
      }),
    ).toEqual(pass);
  });

  it('passes an unknown event through', () => {
    expect(
      run({
        GUARD_EVENT_NAME: 'repository_dispatch',
        GUARD_REF: 'refs/heads/browser-extension',
        GUARD_HEAD_REF: 'browser-extension',
        GUARD_ENVIRONMENT: 'prod',
      }),
    ).toEqual(pass);
  });

  it('ignores the reserved GITHUB_EVENT_NAME', () => {
    expect(
      run({
        GITHUB_EVENT_NAME: 'workflow_dispatch',
        GUARD_REF: 'refs/heads/browser-extension',
        GUARD_ENVIRONMENT: 'dev',
      }),
    ).toEqual(pass);
  });

  it('treats missing variables as empty without crashing', () => {
    expect(run({})).toEqual(pass);
    expect(run({ GUARD_EVENT_NAME: 'pull_request' })).toEqual(pass);
    expect(run({ GUARD_EVENT_NAME: 'workflow_dispatch' })).toEqual(
      fail(NOT_BRANCH),
    );
    expect(
      run({
        GUARD_EVENT_NAME: 'workflow_dispatch',
        GUARD_REF: 'refs/heads/develop',
      }),
    ).toEqual(fail(ENVIRONMENT));
  });

  it('never echoes an input value', () => {
    const results = [
      dispatch('refs/heads/feature/MARKER-7f3a', 'dev'),
      dispatch('refs/heads/feature/MARKER-7f3a', 'prod'),
      dispatch('refs/tags/MARKER-tag-4d2c', 'preview'),
      dispatch('refs/heads/develop', 'MARKER-env-9c1d'),
      dispatch('refs/heads/browser-extension', 'MARKER-env-9c1d'),
      run({
        GUARD_EVENT_NAME: 'pull_request',
        GUARD_REF: 'refs/pull/MARKER-ref-2b8e/merge',
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
  'GUARD_EVENT_NAME: ${{ github.event_name }}',
  'GUARD_REF: ${{ github.ref }}',
  'GUARD_HEAD_REF: ${{ github.head_ref }}',
];

const expectGuardStep = (step: string, scriptPath: string) => {
  expect(step).toContain('name: Guard deploy ref');
  for (const line of GUARD_ENV) expect(step).toContain(line);
  expect(step).not.toContain('GITHUB_EVENT_NAME');
  expect(step).toContain(`run: bash ${scriptPath}\n`);
  // The run line takes no expression: values come through env only.
  const runLine = step.split('\n').find((l) => l.trim().startsWith('run:'));
  expect(runLine).not.toContain('${{');
};

describe('deploy.yml guard', () => {
  const deploy = read('.github/workflows/deploy.yml');
  const steps = jobSteps(deploy, 'setup');

  /** A job's header (everything before its steps). */
  const jobHeader = (job: string) => {
    const start = deploy.indexOf(`\n  ${job}:\n`);
    expect(start, `job ${job} exists`).toBeGreaterThan(-1);
    return deploy.slice(start, deploy.indexOf('\n    steps:\n', start));
  };

  it('gives setup read-only contents', () => {
    expect(jobHeader('setup')).toContain(
      '\n    permissions:\n      contents: read\n',
    );
  });

  it('makes build and terraform wait on setup', () => {
    expect(jobHeader('build')).toContain('\n    needs: setup\n');
    expect(jobHeader('terraform')).toContain('\n    needs: [setup, build]\n');
  });

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
  const cleanup = read('.github/workflows/cleanup-preview.yml');
  const steps = jobSteps(cleanup, 'cleanup');

  it('skips pull requests whose base is not develop or main', () => {
    const job = cleanup.slice(
      cleanup.indexOf('\n  cleanup:\n'),
      cleanup.indexOf('\n    steps:\n'),
    );
    expect(job).toContain(
      "\n    if: github.event_name == 'workflow_dispatch' || github.event.pull_request.base.ref == 'develop' || github.event.pull_request.base.ref == 'main'\n",
    );
    expect(cleanup).toContain('  workflow_dispatch:\n');
  });

  it('runs the guard right after checkout, before any cloud step', () => {
    expect(steps[0]).toContain('uses: actions/checkout@v4');
    expectGuardStep(steps[1], 'scripts/guard-deploy-ref.sh');
    expect(steps[1]).toContain('GUARD_ENVIRONMENT: preview\n');
    expect(steps[2]).toContain('name: Authenticate to Google Cloud');
    const guardAt = steps.findIndex((s) => s.includes('Guard deploy ref'));
    const cloudAt = steps.findIndex((s) =>
      /google-github-actions|gcloud|gsutil|terraform/.test(s),
    );
    expect(guardAt).toBeGreaterThan(-1);
    expect(guardAt).toBeLessThan(cloudAt);
  });
});

describe('a failed guard stops every later step', () => {
  it.each(['deploy.yml', 'promote.yml', 'cleanup-preview.yml'])(
    '%s has no always(), !cancelled() or failure()',
    (file) => {
      const text = read(`.github/workflows/${file}`);
      expect(text).not.toContain('always()');
      expect(text).not.toContain('cancelled()');
      expect(text).not.toContain('failure()');
    },
  );
});
