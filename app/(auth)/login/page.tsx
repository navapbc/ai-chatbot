'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useState, useEffect, Suspense } from 'react';
import { toast } from '@/components/toast';
import { signIn } from 'next-auth/react';
import { MicrosoftLogo } from '@/components/icons/MicrosoftLogo';
import { GoogleLogo } from '@/components/icons/GoogleLogo';

// Check if preview auth mode is enabled via environment variable
const isPreviewAuthModeEnv = process.env.NEXT_PUBLIC_PREVIEW_AUTH_MODE === 'true';

function ErrorHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const error = searchParams.get('error');
    if (error) {
      toast({
        type: 'error',
        description: 'Access denied',
      });
      // Clear the error from URL without refresh
      router.replace('/login', { scroll: false });
    }
  }, [searchParams, router]);

  return null;
}

function LoginContent() {
  const searchParams = useSearchParams();
  const [loadingMethod, setLoadingMethod] = useState<'microsoft' | 'google' | 'credentials' | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // Use callbackUrl from URL params, default to /home
  const callbackUrl = searchParams.get('callbackUrl') || '/home';

  // Check if preview auth mode is enabled via env var OR url param (?previewAuth=true)
  const previewAuthParam = searchParams.get('previewAuth') === 'true';
  const isPreviewAuthMode = isPreviewAuthModeEnv || previewAuthParam;

  const handleGoogleLogin = async () => {
    setLoadingMethod('google');
    try {
      await signIn('google', { callbackUrl });
    } catch (error) {
      toast({
        type: 'error',
        description: 'Failed to sign in with Google',
      });
      setLoadingMethod(null);
    }
  };

  const handleMicrosoftLogin = async () => {
    setLoadingMethod('microsoft');
    try {
      await signIn('microsoft-entra-id', { callbackUrl });
    } catch (error) {
      toast({
        type: 'error',
        description: 'Failed to sign in with Microsoft',
      });
      setLoadingMethod(null);
    }
  };

  const handlePreviewLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      toast({
        type: 'error',
        description: 'Please enter both email and password',
      });
      return;
    }
    setLoadingMethod('credentials');
    try {
      const result = await signIn('credentials', {
        email,
        password,
        previewAuth: previewAuthParam ? 'true' : 'false',
        callbackUrl,
        redirect: false,
      });
      if (result?.error) {
        toast({
          type: 'error',
          description: 'Failed to sign in',
        });
        setLoadingMethod(null);
      } else if (result?.ok) {
        // Use callbackUrl directly to avoid NEXTAUTH_URL issues
        window.location.href = callbackUrl;
      }
    } catch (error) {
      toast({
        type: 'error',
        description: 'Failed to sign in',
      });
      setLoadingMethod(null);
    }
  };

  // Preview auth mode: show email/password form
  if (isPreviewAuthMode) {
    return (
      <div className="bg-chat-background relative size-full min-h-screen">
        <div className="absolute bg-card border border-border border-solid left-1/2 rounded-[10px] top-[200px] -translate-x-1/2 w-[414px] p-8">
          <div className="flex flex-col gap-4 items-center">
            <p className="font-source-serif leading-normal not-italic text-[32px] text-center text-card-foreground tracking-[0.16px]">
              Welcome
            </p>
            <p className="font-inter font-normal leading-normal not-italic text-[14px] text-center text-muted-foreground tracking-[0.07px]">
              Sign in to access the Form-Filling Assistant
            </p>
            <span className="inline-block bg-yellow-100 text-yellow-800 px-3 py-1 rounded text-xs font-semibold">
              Preview Environment
            </span>

            <form onSubmit={handlePreviewLogin} className="w-full flex flex-col gap-4 mt-2">
              <div className="flex flex-col gap-1.5">
                <label htmlFor="email" className="text-sm font-medium text-card-foreground">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter any email"
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-card-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  required
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label htmlFor="password" className="text-sm font-medium text-card-foreground">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter any password"
                  className="w-full px-3 py-2 border border-border rounded-md bg-background text-card-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  required
                />
              </div>
              <button
                type="submit"
                disabled={loadingMethod !== null}
                className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-md font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loadingMethod === 'credentials' ? 'Signing in...' : 'Sign In'}
              </button>
            </form>

            <p className="text-xs text-muted-foreground text-center mt-2">
              Enter any email and password to access the preview environment.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Production auth mode: show OAuth buttons
  return (
    <div className="bg-chat-background relative size-full min-h-screen">
      <div className="absolute bg-card border border-border border-solid h-[260px] left-1/2 rounded-[10px] top-[257px] -translate-x-1/2 w-[414px]">
        <div className="absolute content-stretch flex flex-col gap-[18px] h-[42px] items-center left-[32px] top-[32px] w-[350px]">
          <p className="font-source-serif leading-normal min-w-full not-italic relative shrink-0 text-[32px] text-center text-card-foreground tracking-[0.16px]">
            Welcome
          </p>
          <p className="font-inter font-normal leading-normal min-w-full not-italic relative shrink-0 text-[14px] text-center text-muted-foreground tracking-[0.07px]">
            Sign in to access the Form-Filling Assistant
          </p>

          {/* Microsoft Login Button */}
          <button
            type="button"
            onClick={handleMicrosoftLogin}
            disabled={loadingMethod !== null}
            className="border border-border border-solid box-border content-stretch flex gap-[8px] items-center justify-center min-h-[36px] px-[16px] py-[7.5px] relative rounded-[8px] shrink-0 w-full hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors bg-card"
          >
            <div className="relative shrink-0 size-[13.25px]">
              <MicrosoftLogo size={13.25} className="block max-w-none size-full" />
            </div>
            <div className="flex flex-col font-inter font-medium justify-center leading-[0] not-italic relative shrink-0 text-[14px] text-center text-card-foreground text-nowrap">
              <p className="leading-normal whitespace-pre">
                {loadingMethod === 'microsoft' ? 'Signing in...' : 'Continue with Microsoft'}
              </p>
            </div>
          </button>

          {/* Google Login Button */}
          <button
            type="button"
            onClick={handleGoogleLogin}
            disabled={loadingMethod !== null}
            className="border border-border border-solid box-border content-stretch flex gap-[8px] items-center justify-center min-h-[36px] px-[16px] py-[7.5px] relative rounded-[8px] shrink-0 w-full hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors bg-card"
          >
            <div className="relative shrink-0 size-[13.25px]">
              <GoogleLogo size={13.25} className="block max-w-none size-full" />
            </div>
            <div className="flex flex-col font-inter font-medium justify-center leading-[0] not-italic relative shrink-0 text-[14px] text-center text-card-foreground text-nowrap">
              <p className="leading-normal whitespace-pre">
                {loadingMethod === 'google' ? 'Signing in...' : 'Continue with Google'}
              </p>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <ErrorHandler />
      <LoginContent />
    </Suspense>
  );
}
