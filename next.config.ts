import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // cacheComponents disabled to allow runtime env vars in API routes
  // See: https://github.com/vercel/next.js/discussions/84894
  cacheComponents: false,
  serverExternalPackages: ['agent-browser', 'playwright-core'],
  images: {
    remotePatterns: [
      {
        hostname: 'avatar.vercel.sh',
      },
    ],
  },
};

export default nextConfig;
