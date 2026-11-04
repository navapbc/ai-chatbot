import { z } from 'zod';

export type SectionStatus = 'waiting' | 'filling' | 'done' | 'needs_you';

/** A field's state inside a section. The label names the field, never a value. */
export type FieldState = 'waiting' | 'filled' | 'needs_you' | 'skipped';

export interface FieldProgress {
  label: string;
  state: FieldState;
}

export interface RunProgress {
  sections: {
    name: string;
    status: SectionStatus;
    filled: number;
    total?: number;
    fields?: FieldProgress[];
  }[];
  current?: string;
  actions?: string[];
}

const count = z.number().int().min(0);

// z.object strips unknown keys by default, so applicant values the model
// might attach never survive parsing.
export const runProgressSchema: z.ZodType<RunProgress> = z.object({
  sections: z
    .array(
      z.object({
        name: z.string().min(1).max(80),
        status: z.enum(['waiting', 'filling', 'done', 'needs_you']),
        filled: count,
        total: count.optional(),
        // Field labels only (First name, Street address): the same rule as
        // section names. The panel shows them so the caseworker can compare
        // the page against the record as the agent works.
        fields: z
          .array(
            z.object({
              label: z.string().min(1).max(80),
              state: z.enum(['waiting', 'filled', 'needs_you', 'skipped']),
            }),
          )
          .max(30)
          .optional(),
      }),
    )
    .max(40),
  current: z.string().max(120).optional(),
  actions: z.array(z.string().max(120)).max(5).optional(),
});

export function sanitizeProgress(input: unknown): RunProgress | null {
  const parsed = runProgressSchema.safeParse(input);
  return parsed.success ? parsed.data : null;
}
