// Build-time configuration, substituted by scripts/build-extension.ts through
// esbuild's --define. Outside an esbuild bundle (e.g. a unit test importing
// this module) the identifier is undeclared, so fall back to ''.

declare const __APP_ORIGIN__: string;

/** The web app's exact origin, or '' when the build set no APP_ORIGIN. */
export const APP_ORIGIN: string =
  typeof __APP_ORIGIN__ === 'string' ? __APP_ORIGIN__ : '';
