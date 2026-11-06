import Image from 'next/image';

const row1 = [
  { src: '/images/claude.png', alt: 'Claude', className: 'h-12' },
  { src: '/images/vertexai.jpeg', alt: 'Vertex AI', className: 'h-[72px]' },
];

const row2 = [
  { src: '/images/agent-browser.png', alt: 'Agent Browser', className: 'h-12' },
  { src: '/images/aisdk.png', alt: 'AI SDK', className: 'h-36' },
];

const row3 = [
  { src: '/images/logo-kernel-light.svg', alt: 'Kernel', className: 'h-12' },
];

export default function LogoWallPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-white px-6">
      <div className="max-w-3xl text-center">
        <p className="mb-10 text-3xl font-medium uppercase tracking-widest text-neutral-500">
          Built with
        </p>
        <div className="flex flex-col items-center gap-2">
          {[row1, row2, row3].map((row) => (
            <div
              key={row.map((logo) => logo.alt).join('-')}
              className="flex flex-wrap items-center justify-center gap-12"
            >
              {row.map((logo) => (
                <div
                  key={logo.alt}
                  className="flex items-center justify-center grayscale opacity-70 transition hover:opacity-100 hover:grayscale-0"
                >
                  <Image
                    src={logo.src}
                    alt={logo.alt}
                    width={640}
                    height={240}
                    className={`${logo.className} w-auto object-contain`}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
