import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// The panel's copy of `cn` from lib/utils. The panel bundle aliases the exact
// specifier `@/lib/utils` here (scripts/extension-build-config.ts), because
// lib/utils also imports `ai`, the database schema and app types, none of
// which belong in the extension. Same signature as lib/utils's `cn`.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
