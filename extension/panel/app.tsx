import { Button } from '@/components/ui/button';

/** Placeholder side panel: the start flow replaces it. */
export const App = () => (
  <main className="flex min-h-screen flex-col gap-4 bg-background p-4 text-foreground">
    <h1 className="text-lg font-semibold">Labs ASP</h1>
    <Button type="button">Start</Button>
  </main>
);
