// The first message of an extension (relay) run. The marker line tells the
// agent it is driving the caseworker's own browser tab and points it at the
// `extension-run` skill. The caseworker's text follows unchanged, so anything
// that parses it (the participant record) still finds it.
export const EXTENSION_RUN_MARKER = '[extension-run]';

export const buildExtensionRunMessage = (text: string): string =>
  `${EXTENSION_RUN_MARKER}\nThis is an extension run: read the \`extension-run\` skill with \`load_skill\` before you start. It overrides your other instructions where they differ.\n\n${text}`;
