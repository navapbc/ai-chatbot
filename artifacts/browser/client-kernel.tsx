'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { MousePointerClick, RefreshCw, Monitor } from 'lucide-react';
import { toast } from 'sonner';
import { AgentStatusIndicator } from '@/components/agent-status-indicator';
import { BrowserLoadingState, BrowserErrorState } from './browser-states';
import { useIsMobile } from '@/hooks/use-mobile';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import type { ChatStatus } from '@/components/create-artifact';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { ChatMessage } from '@/lib/types';

interface KernelBrowserClientProps {
  sessionId: string;
  liveViewUrl?: string;
  controlMode: 'agent' | 'user';
  onControlModeChange: (mode: 'agent' | 'user') => void;
  onConnectionChange?: (connected: boolean) => void;
  chatStatus?: ChatStatus;
  stop?: () => void;
  isFullscreen?: boolean;
  onFullscreenChange?: (fullscreen: boolean) => void;
  sendMessage?: UseChatHelpers<ChatMessage>['sendMessage'];
}

export function KernelBrowserClient({
  sessionId,
  liveViewUrl: liveViewUrlProp,
  controlMode,
  onControlModeChange,
  onConnectionChange,
  chatStatus,
  stop,
  isFullscreen = false,
  onFullscreenChange,
  sendMessage,
}: KernelBrowserClientProps) {
  const [error, setError] = useState<string | null>(null);
  const [isSheetOpen, setIsSheetOpen] = useState(false);
  const isMobile = useIsMobile();

  // Use refs to avoid dependency changes triggering re-renders in effects
  const onConnectionChangeRef = useRef(onConnectionChange);
  onConnectionChangeRef.current = onConnectionChange;

  // Keep sessionId in a ref so the beforeunload handler always has the latest value
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  // On mount, if no liveViewUrl was provided via stream (e.g. page refresh),
  // try to recover it from the server's in-memory session cache. Single
  // request, no polling — if the session isn't there, the browser is gone.
  const [recoveredUrl, setRecoveredUrl] = useState<string | null>(null);
  useEffect(() => {
    if (liveViewUrlProp || recoveredUrl) return;

    let cancelled = false;
    (async () => {
      try {
        console.log('[KernelBrowserClient] Recovery: fetching existing session for', sessionId);
        const res = await fetch('/api/kernel-browser', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'get', sessionId }),
        });
        const data = await res.json();
        console.log('[KernelBrowserClient] Recovery response:', res.status, data);
        if (res.ok && data.liveViewUrl && !cancelled) {
          setRecoveredUrl(data.liveViewUrl);
        }
      } catch (err) {
        console.error('[KernelBrowserClient] Recovery failed:', err);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId, liveViewUrlProp, recoveredUrl]);

  const effectiveUrl = liveViewUrlProp || recoveredUrl;
  const isConnected = !!effectiveUrl;

  // Notify parent when connection state changes
  const prevConnectedRef = useRef(false);
  useEffect(() => {
    if (isConnected !== prevConnectedRef.current) {
      prevConnectedRef.current = isConnected;
      onConnectionChangeRef.current?.(isConnected);
    }
  }, [isConnected]);

  // Listen for control mode switch events from confirmation components
  useEffect(() => {
    const handleSwitchControl = (event: CustomEvent) => {
      const { mode } = event.detail;
      if (mode === 'user' || mode === 'agent') {
        switchControlMode(mode);
      }
    };

    window.addEventListener('switch-browser-control', handleSwitchControl as EventListener);

    return () => {
      window.removeEventListener('switch-browser-control', handleSwitchControl as EventListener);
    };
  }, []);

  // Global keyboard listener for fullscreen mode - Escape to exit
  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isFullscreen && controlMode === 'user') {
        event.preventDefault();
        switchControlMode('agent');
      }
    };

    if (isFullscreen && controlMode === 'user') {
      document.addEventListener('keydown', handleGlobalKeyDown);
      return () => document.removeEventListener('keydown', handleGlobalKeyDown);
    }
  }, [isFullscreen, controlMode]);

  const switchControlMode = (mode: 'agent' | 'user') => {
    if (!isConnected) {
      toast.error('Not connected to browser session');
      return;
    }

    console.log(`[Kernel] Switching control mode to: ${mode}`);

    if (mode === 'user') {
      // Stop the AI when user takes control
      if (stop) {
        stop();
      }
      // On desktop, automatically enable fullscreen when switching to user mode
      if (!isMobile) {
        onFullscreenChange?.(true);
      }
    } else {
      // Exit fullscreen when giving back control to agent
      onFullscreenChange?.(false);

      // Send a message to the agent so it knows to snapshot and continue
      if (sendMessage) {
        sendMessage({
          role: 'user' as const,
          parts: [{
            type: 'text' as const,
            text: "I've finished making my changes to the page. Please take a snapshot to review what I updated and continue from where you left off.",
          }],
        });
      }
    }

    onControlModeChange(mode);
  };

  // Build the iframe URL with readOnly based on control mode
  const iframeUrl = useMemo(() => {
    if (!effectiveUrl) return null;

    const url = new URL(effectiveUrl);
    if (controlMode === 'agent') {
      url.searchParams.set('readOnly', 'true');
    } else {
      url.searchParams.delete('readOnly');
    }
    return url.toString();
  }, [effectiveUrl, controlMode]);

  if (error) {
    return <BrowserErrorState onRetry={() => setError(null)} />;
  }

  if (!effectiveUrl) {
    return <BrowserLoadingState />;
  }

  // Fullscreen mode when user has control
  if (controlMode === 'user' && isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 browser-fullscreen-bg flex flex-col overflow-hidden">
        {/* Fullscreen header with controls */}
        <div className="sticky top-0 left-0 right-0 z-10 browser-fullscreen-bg">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between px-2 sm:px-4 py-2 sm:py-3 gap-2">
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <div className="size-2 bg-red-500 rounded-full animate-pulse status-indicator" />
                <span className="text-xs sm:text-sm font-medium font-ibm-plex-mono browser-fullscreen-text">You're editing manually</span>
              </div>
              <span className="text-xs sm:text-sm browser-fullscreen-text font-inter hidden sm:block">
                The AI will continue with your changes when you give back control.
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="default"
                size="sm"
                onClick={() => switchControlMode('agent')}
              >
                Give back control
              </Button>
            </div>
          </div>
        </div>

        {/* Fullscreen browser iframe */}
        <div className="flex-1 overflow-hidden browser-fullscreen-bg pt-20 pb-4 sm:pb-12 px-2 sm:px-4 md:px-12">
          <div className="w-full h-full flex items-center justify-center">
            <iframe
              key={effectiveUrl}
              src={iframeUrl || undefined}
              className="border-0 bg-white rounded-lg shadow-2xl"
              style={{
                width: '1280px',
                height: '800px',
                maxWidth: '100%',
                maxHeight: '100%',
              }}
              allow="autoplay; clipboard-read; clipboard-write"
              title="Browser View"
            />
          </div>
        </div>
      </div>
    );
  }

  // Mobile drawer mode
  if (isMobile) {
    return (
      <div className="pointer-events-none">
        {/* Mobile: Floating button to open browser drawer */}
        <div className="fixed top-4 right-4 z-[100] pointer-events-auto">
          <Button
            type="button"
            variant="default"
            size="lg"
            onClick={() => setIsSheetOpen(true)}
            className="rounded-full shadow-lg"
          >
            <Monitor className="w-5 h-5 mr-2" />
            View Browser
          </Button>
        </div>

        {/* Mobile: Bottom sheet with browser content */}
        <div className="pointer-events-auto">
          <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
            <SheetContent
              side="bottom"
              className="h-[85vh] p-0 overflow-y-scroll flex flex-col z-[100]"
            >
              <SheetHeader className="px-4 py-3 border-b">
                <SheetTitle className="text-left">Browser View</SheetTitle>
              </SheetHeader>

              {/* Control mode indicator */}
              {isConnected && (
                <div className="flex-shrink-0 flex items-center justify-between py-2 px-4 bg-muted/20">
                  <AgentStatusIndicator
                    chatStatus={chatStatus}
                    controlMode={controlMode}
                  />
                  <Button
                    type="button"
                    variant="default"
                    size="sm"
                    onClick={() => switchControlMode(controlMode === 'user' ? 'agent' : 'user')}
                  >
                    {controlMode === 'user' ? (
                      'Give back control'
                    ) : (
                      <>
                        <MousePointerClick className="w-4 h-4 mr-1" />
                        Take control
                      </>
                    )}
                  </Button>
                </div>
              )}

              {/* Browser content */}
              <div className="flex-1 overflow-hidden min-h-0 p-4">
                <div className={`h-full overflow-hidden bg-black rounded-lg ${controlMode === 'agent' ? 'cursor-not-allowed' : 'cursor-auto'}`}>
                  <iframe
                    key={effectiveUrl}
                    src={iframeUrl || undefined}
                    className="w-full h-full border-0 bg-white shadow-lg"
                    allow="autoplay; clipboard-read; clipboard-write"
                    title="Browser View"
                  />
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    );
  }

  // Normal (non-fullscreen) desktop mode
  return (
    <div className="h-full flex flex-col">
      {/* Control mode indicator and buttons */}
      {isConnected && (
        <div className="flex-shrink-0 flex items-center justify-between py-2 bg-muted/20">
          <AgentStatusIndicator
            chatStatus={chatStatus}
            controlMode={controlMode}
            className="text-sm text-black"
          />
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="default"
              size="sm"
              onClick={() => switchControlMode(controlMode === 'user' ? 'agent' : 'user')}
            >
              <MousePointerClick className="w-4 h-4" />
              {controlMode === 'user' ? 'Give back control' : 'Take control'}
            </Button>
          </div>
        </div>
      )}

      {/* Browser iframe - fixed pixel dimensions to prevent layout recalculation flicker */}
      <div className={`flex-1 overflow-hidden m-4 min-h-0 flex items-center justify-center ${controlMode === 'agent' ? 'cursor-not-allowed' : 'cursor-auto'}`}>
        <iframe
          key={effectiveUrl}
          src={iframeUrl || undefined}
          className="border-0 bg-white rounded-lg"
          style={{
            width: '1280px',
            height: '800px',
            maxWidth: '100%',
            maxHeight: '100%',
            pointerEvents: controlMode === 'agent' ? 'none' : 'auto',
          }}
          allow="autoplay; clipboard-read; clipboard-write"
          title="Browser View"
        />
      </div>
    </div>
  );
}
