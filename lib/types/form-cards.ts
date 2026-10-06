// Shared types for the gap-analysis and form-summary cards.
// Tools emit a flat ordered field list; this module chunks that list
// into pages of PAGE_SIZE so the modal can paginate. Old chats that
// still carry a `sections` shape are flattened in order then re-chunked,
// so original section titles are discarded.

export type GapField = {
  field: string;
  options?: string[];
  inputType?: 'text' | 'select' | 'date' | 'boolean' | 'textarea';
  multiSelect?: boolean;
  condition?: string;
  required?: boolean;
  placeholder?: string;
  note?: string;
};

/**
 * `record` = the participant JSON the caseworker pasted into the opening
 * message. It is the canonical token everywhere downstream of
 * `adaptReviewSections`.
 *
 * The legacy route's tool still emits `database`, a name left over from the
 * removed Apricot client-database integration — there is no database now, the
 * record simply arrives inline (see lib/data/participants.ts and
 * buildApplicationPrompt). `database` is load-bearing in the Braintrust eval
 * suite, its golden datasets and the online scorers, so it is NOT renamed
 * there; it is normalised to `record` on the way into the card instead.
 */
export type FieldSource = 'record' | 'caseworker' | 'inferred' | 'missing';

/** What a tool may emit, before normalisation. */
export type RawFieldSource = FieldSource | 'database';

export type ReviewField = {
  field: string;
  value?: string;
  source: FieldSource;
  inputType?: 'text' | 'select' | 'radio' | 'checkbox';
  options?: string[];
  required?: boolean;
  inferredFrom?: string;
};

export type GapSection = {
  id: string;
  title: string;
  fields: GapField[];
};

export type ReviewSection = {
  id: string;
  title: string;
  fields: ReviewField[];
};

const PAGE_SIZE = 5;

type LegacyGapInput = {
  sections?: GapSection[];
  missingFields?: GapField[];
};

type RawReviewField = Omit<ReviewField, 'source'> & { source: RawFieldSource };

type LegacyReviewInput = {
  sections?: RawReviewSection[];
  fields?: RawReviewField[];
};

type RawReviewSection = Omit<ReviewSection, 'fields'> & {
  fields: RawReviewField[];
};

/**
 * Single normalisation point for the source token. Every review field reaches
 * the card through `adaptReviewSections`, so aliasing here keeps `database`
 * out of the UI types and out of every consumer — rather than each branch
 * having to remember both spellings and silently falling through to "Manual"
 * when it forgets.
 */
function normalizeField(f: RawReviewField): ReviewField {
  return { ...f, source: f.source === 'database' ? 'record' : f.source };
}

function chunk<T>(items: T[], size: number): T[][] {
  if (items.length === 0) return [];
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

export function adaptGapSections(
  input: LegacyGapInput | undefined,
): GapSection[] {
  if (!input) return [];
  const flat: GapField[] = input.missingFields?.length
    ? input.missingFields
    : (input.sections ?? []).flatMap((s) => s.fields);
  return chunk(flat, PAGE_SIZE).map((fields, i) => ({
    id: `page-${i}`,
    title: '',
    fields,
  }));
}

export function adaptReviewSections(
  input: LegacyReviewInput | undefined,
): ReviewSection[] {
  if (!input) return [];
  const flat: ReviewField[] = (
    input.fields?.length
      ? input.fields
      : (input.sections ?? []).flatMap((s) => s.fields)
  ).map(normalizeField);
  return chunk(flat, PAGE_SIZE).map((fields, i) => ({
    id: `page-${i}`,
    title: '',
    fields,
  }));
}
