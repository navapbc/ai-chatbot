'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ConsentPage } from '@/components/consent-page';

export default function ConsentRoutePage() {
  const router = useRouter();
  const [hasConsented, setHasConsented] = useState(false);

  const handleConsent = () => {
    setHasConsented(true);
    // Redirect to the main chat after consent
    router.push('/chat');
  };

  return <ConsentPage onConsent={handleConsent} />;
}
