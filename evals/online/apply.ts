// Applies the online-eval config to one environment's Braintrust project.
// Idempotent; rules stay paused unless --activate.
//
//   pnpm eval:online:apply dev [--activate] [--dry-run]
//
// Project names mirror BRAINTRUST_PARENT in terraform/cloud_run.tf. Keep in
// sync with terraform's `environments` map.

import {
  BTQL_FILTER,
  EVALUATOR_DEFINITION,
  EVALUATOR_SLUG,
  RULE_NAME,
  RULE_SCOPE,
} from './gap-analysis-asking';

const ENVIRONMENTS = ['dev', 'preview', 'prod'] as const;
type Environment = (typeof ENVIRONMENTS)[number];

const API = 'https://api.braintrust.dev';

const apiKey = process.env.BRAINTRUST_API_KEY;
if (!apiKey) {
  throw new Error('BRAINTRUST_API_KEY is not set');
}

const args = process.argv.slice(2);
const env = args.find((a) => !a.startsWith('--')) as Environment | undefined;
const activate = args.includes('--activate');
const dryRun = args.includes('--dry-run');

if (!env || !ENVIRONMENTS.includes(env)) {
  throw new Error(
    `Usage: apply.ts <${ENVIRONMENTS.join('|')}> [--activate] [--dry-run]`,
  );
}

const projectName = `labs-asp-${env}`;

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

const main = async () => {
  // Braintrust creates projects on first trace, so prod may not exist yet.
  const { objects: projects } = await request(
    'GET',
    `/v1/project?project_name=${encodeURIComponent(projectName)}`,
  );
  const project = projects?.[0];
  if (!project) {
    throw new Error(
      `Project "${projectName}" does not exist. Braintrust creates it on the ` +
        `first exported trace — deploy ${env} and send one request first.`,
    );
  }

  if (dryRun) {
    console.log(
      JSON.stringify(
        {
          project: projectName,
          project_id: project.id,
          evaluator: EVALUATOR_SLUG,
          rule: RULE_NAME,
          status: activate ? 'active' : 'paused',
        },
        null,
        2,
      ),
    );
    return;
  }

  // Upsert by slug.
  const evaluator = await request('PUT', '/v1/function', {
    project_id: project.id,
    function_type: 'scorer',
    ...EVALUATOR_DEFINITION,
  });

  // PUT replaces by name; POST no-ops on an existing rule and leaves it stale.
  const rule = await request('PUT', '/v1/project_score', {
    project_id: project.id,
    name: RULE_NAME,
    score_type: 'online',
    description: `Scores caseworker-facing gapAnalysis quality on ${env} traffic.`,
    config: {
      online: {
        sampling_rate: 1,
        scorers: [{ type: 'function', id: evaluator.id }],
        btql_filter: BTQL_FILTER,
        scope: RULE_SCOPE,
        status: activate ? 'active' : 'paused',
      },
    },
  });

  console.log(
    JSON.stringify(
      {
        project: projectName,
        evaluator_id: evaluator.id,
        rule_id: rule.id,
        status: activate ? 'active' : 'paused',
      },
      null,
      2,
    ),
  );
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
