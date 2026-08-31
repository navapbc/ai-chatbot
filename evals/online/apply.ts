// Applies the online-eval config to one environment's Braintrust project.
// Idempotent; rules stay paused unless --activate.
//
//   pnpm eval:online:apply dev [--activate] [--dry-run]
//
// Project names mirror BRAINTRUST_PARENT in terraform/cloud_run.tf. Keep in
// sync with terraform's `environments` map.

import { RULES } from './rules';

const ENVIRONMENTS = ['dev', 'preview', 'prod'] as const;
type Environment = (typeof ENVIRONMENTS)[number];

const API = 'https://api.braintrust.dev';

const apiKey = process.env.BRAINTRUST_API_KEY;
if (!apiKey) {
  throw new Error('BRAINTRUST_API_KEY is not set');
}

const args = process.argv.slice(2);
const target = args.find((a) => !a.startsWith('--'));
const activate = args.includes('--activate');
const dryRun = args.includes('--dry-run');

if (
  !target ||
  (target !== 'all' && !ENVIRONMENTS.includes(target as Environment))
) {
  throw new Error(
    `Usage: apply.ts <${ENVIRONMENTS.join('|')}|all> [--activate] [--dry-run]`,
  );
}

const targets: Environment[] =
  target === 'all' ? [...ENVIRONMENTS] : [target as Environment];

const request = async (
  method: string,
  path: string,
  body?: unknown,
): Promise<any> => {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  if (!res.ok) {
    throw new Error(
      `${method} ${path} failed (${res.status}): ${await res.text()}`,
    );
  }
  return res.json();
};

const applyEnv = async (env: Environment) => {
  const projectName = `labs-asp-${env}`;

  // Braintrust creates projects on first trace, so prod may not exist yet.
  const { objects: projects } = await request(
    'GET',
    `/v1/project?project_name=${encodeURIComponent(projectName)}`,
  );
  const project = projects?.[0];
  if (!project) {
    return {
      project: projectName,
      skipped: `does not exist yet — send one trace from ${env} first`,
    };
  }

  if (dryRun) {
    return {
      project: projectName,
      project_id: project.id,
      rules: RULES.map((r) => ({
        rule: r.name,
        scorers: r.scorers.map((sc) => sc.slug),
        status: activate ? 'active' : 'paused',
      })),
    };
  }

  const applied: unknown[] = [];
  for (const rule of RULES) {
    // Upsert each judge by slug, then point the rule at all of them.
    const scorerIds: string[] = [];
    for (const scorer of rule.scorers) {
      const fn = await request('PUT', '/v1/function', {
        project_id: project.id,
        function_type: 'scorer',
        ...scorer,
      });
      scorerIds.push(fn.id);
    }

    // PUT replaces by name; POST no-ops on an existing rule and leaves it stale.
    const saved = await request('PUT', '/v1/project_score', {
      project_id: project.id,
      name: rule.name,
      score_type: 'online',
      description: `${rule.description} (${env})`,
      config: {
        online: {
          sampling_rate: rule.samplingRate,
          scorers: scorerIds.map((id) => ({ type: 'function', id })),
          btql_filter: rule.btqlFilter,
          scope: rule.scope,
          status: activate ? 'active' : 'paused',
        },
      },
    });

    applied.push({
      rule: rule.name,
      rule_id: saved.id,
      scorers: rule.scorers.map((sc, i) => ({
        slug: sc.slug,
        id: scorerIds[i],
      })),
      status: activate ? 'active' : 'paused',
    });
  }

  return { project: projectName, rules: applied };
};

const main = async () => {
  const results = [];
  for (const env of targets) {
    results.push(await applyEnv(env));
  }
  console.log(JSON.stringify(results, null, 2));
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
