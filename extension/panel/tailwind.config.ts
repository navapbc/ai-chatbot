import path from 'node:path';
import type { Config } from 'tailwindcss';
import base from '../../tailwind.config';

// The side panel's Tailwind config: the app's theme (colors, radius, dark
// mode, plugins) with content limited to the panel and the components/ui leaf
// files it renders. Content globs are absolute because Tailwind 3 resolves
// relative ones against the working directory, not this file.
const ROOT = path.resolve(__dirname, '../..');

const config: Config = {
  ...base,
  content: [
    path.join(ROOT, 'extension/panel/**/*.{ts,tsx,html}'),
    path.join(ROOT, 'components/ui/**/*.{ts,tsx}'),
  ],
};

export default config;
