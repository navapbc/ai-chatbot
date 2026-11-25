import { z } from 'zod';
export const Preparation = z
  .object({
    requestId: z.uuid(),
    applicationId: z.uuid(),
    expectedVersion: z.number().int().positive(),
    section: z.enum([
      'personal',
      'expedited',
      'household',
      'additional',
      'income',
      'resources',
      'expenses',
    ]),
    memberId: z.string().max(100).optional(),
    data: z.record(z.string(), z.unknown()),
    guide: z.unknown(),
  })
  .strict();
export type Preparation = z.infer<typeof Preparation>;
