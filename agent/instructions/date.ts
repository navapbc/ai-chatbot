import { defineDynamic, defineInstructions } from 'eve/instructions';

// Runtime date must resolve per session, not at build time — a plain
// defineInstructions module is captured once at build. defineDynamic's
// session.started resolver runs at session start, so the date is fresh.
export default defineDynamic({
  events: {
    'session.started': () => {
      const now = new Date();
      const formatted = now.toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
      const iso = now.toISOString().split('T')[0];
      return defineInstructions({
        markdown: `Today's date is ${formatted} (${iso}). Use this date for any age calculations, "today's date" fields, or date-relative logic.`,
      });
    },
  },
});
