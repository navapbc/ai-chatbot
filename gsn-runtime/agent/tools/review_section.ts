import { defineTool } from 'eve/tools';
import { choice } from '@typesafe-ai/sdk';
import { z } from 'zod';
import { askJev, isJevEnabled } from '../../../lib/jev/client';
import { requestState } from '../lib/request';
export default defineTool({
  description:
    'Return the exact resident-provided section proposal with a fixed-question Jev review. Does not save, sign or submit.',
  inputSchema: z.object({ summary: z.string().min(1).max(2000) }),
  async execute({ summary }, ctx) {
    const request = requestState.get();
    if (!request || !isJevEnabled('summary-check'))
      throw new Error('Combined preparation is not configured.');
    const result = await askJev({
      state: {
        resident_provided_answers: request.data,
        section: request.section,
        memberId: request.memberId ?? null,
        application_guide: request.guide,
      },
      questions: {
        fit: choice(
          'Do these resident-provided answers appear appropriate for the named application section and guide? Do not infer eligibility or treat missing answers as No.',
          {
            consistent: 'The supplied answers fit the question meanings.',
            needs_review:
              'One or more answers or question meanings need resident review.',
          },
        ),
      },
      feature: 'summary-check',
      timeoutMs: 15000,
      signal: ctx.abortSignal,
    });
    if (!result.ok)
      throw new Error('Jev did not complete the preparation check.');
    return {
      format: 'gsn-eve-jev-preparation-v1',
      requestId: request.requestId,
      section: request.section,
      memberId: request.memberId,
      data: request.data,
      summary,
      stopBefore: 'signature-and-submission',
      jev: {
        mode: 'live',
        model: result.model,
        answers: result.answers,
        usage: result.usage,
        policy: 'annotate-only; resident review required',
      },
    };
  },
});
