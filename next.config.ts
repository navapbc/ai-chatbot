import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // cacheComponents disabled to allow runtime env vars in API routes
  // See: https://github.com/vercel/next.js/discussions/84894
  cacheComponents: false,
  // agent-browser is a native binary invoked as a subprocess, not an imported
  // module, so there is nothing for Next.js to bundle or externalize.
  images: {
    remotePatterns: [
      {
        hostname: 'avatar.vercel.sh',
      },
    ],
  },
};

export default nextConfig;
