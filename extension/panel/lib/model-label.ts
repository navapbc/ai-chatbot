// Display names for model ids in the picker: "claude-opus-5" reads as
// "Claude Opus 5". Unknown shapes fall back to the id itself.
const CLAUDE_ID = /^claude-(opus|sonnet|haiku)-(\d+(?:-\d+)?)$/;
const GPT_ID = /^gpt-(\d+(?:\.\d+)?)(?:-(pro|mini|nano))?$/;

const capitalize = (word: string) => word[0].toUpperCase() + word.slice(1);

export const modelLabel = (id: string): string => {
  const claude = CLAUDE_ID.exec(id);
  if (claude)
    return `Claude ${capitalize(claude[1])} ${claude[2].replace('-', '.')}`;
  const gpt = GPT_ID.exec(id);
  if (gpt) return `GPT-${gpt[1]}${gpt[2] ? ` ${capitalize(gpt[2])}` : ''}`;
  return id;
};
