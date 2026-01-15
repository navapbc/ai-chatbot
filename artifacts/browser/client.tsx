import { Artifact, type ChatContext } from '@/components/create-artifact';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { MonitorX, Loader2, RefreshCwIcon, Monitor, MousePointerClick, ClockFading } from 'lucide-react';
import { toast } from 'sonner';
import { useIsMobile } from '@/hooks/use-mobile';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { AgentStatusIndicator } from '@/components/agent-status-indicator';
import { useDataStream } from '@/components/data-stream-provider';

interface BrowserArtifactMetadata {
  sessionId: string;
  isConnected: boolean;
  isConnecting: boolean;
  connectionUrl?: string;
  error?: string;
  controlMode: 'agent' | 'user';
  isFullscreen: boolean;
  isSheetOpen?: boolean;
  setIsSheetOpen?: (open: boolean) => void;
  // Browserbase specific
  liveViewUrl?: string;
}

export const browserArtifact = new Artifact<'browser', BrowserArtifactMetadata>({
  kind: 'browser',
  description: 'Live browser automation display with Browserbase Live View',

  initialize: async ({ documentId, setMetadata, chatContext }) => {
    // Use chat session ID (threadId-resourceId) for browser session isolation
    // This ensures each chat gets its own browser session
    let sessionId: string;

    if (chatContext?.chatId && chatContext?.resourceId) {
      sessionId = `${chatContext.chatId}-${chatContext.resourceId}`;
      console.log(`[Browser Artifact] Using chat session ID: ${sessionId}`);
    } else {
      sessionId = `browser-${documentId}-${Date.now()}`;
      console.warn(`[Browser Artifact] No chat context, using document-based session ID: ${sessionId}`);
    }

    setMetadata({
      sessionId,
      isConnected: false,
      isConnecting: false,
      controlMode: 'agent',
      isFullscreen: false,
    });
  },

  onStreamPart: ({ streamPart, setMetadata, setArtifact }) => {
    // Handle artifact creation - make it visible when streaming starts
    if (streamPart.type === 'data-kind' && streamPart.data === 'browser') {
      setArtifact((draftArtifact) => ({
        ...draftArtifact,
        isVisible: true,
        status: 'streaming',
      }));
    }

    // Handle content updates
    if (streamPart.type === 'data-textDelta') {
      setArtifact((draftArtifact) => ({
        ...draftArtifact,
        content: draftArtifact.content + streamPart.data,
        status: 'streaming',
      }));
    }
  },

  content: ({ metadata, setMetadata, isCurrentVersion, status, chatStatus, stop }) => {
    const isMobile = useIsMobile();

    // Get liveViewUrl from data stream (sent by backend when using Browserbase)
    const { liveViewUrl: streamedLiveViewUrl } = useDataStream();

    // When liveViewUrl comes from the stream, update metadata and mark connected
    useEffect(() => {
      if (streamedLiveViewUrl && metadata && !metadata.liveViewUrl) {
        console.log('[Browser] Received liveViewUrl from stream:', streamedLiveViewUrl);
        setMetadata({
          ...metadata,
          isConnected: true,
          isConnecting: false,
          liveViewUrl: streamedLiveViewUrl,
          connectionUrl: streamedLiveViewUrl,
        });
        toast.success('Connected to Browserbase');
      }
    }, [streamedLiveViewUrl, metadata, setMetadata]);

    // Auto-connect: set connecting state when artifact becomes visible
    useEffect(() => {
      if (metadata && !metadata.isConnected && !metadata.isConnecting && !metadata.liveViewUrl) {
        // The liveViewUrl will come from the backend stream
        setMetadata({
          ...metadata,
          isConnecting: true,
        });
      }
    }, [isCurrentVersion, metadata?.sessionId]);

    // Switch control mode between agent and user
    const switchControlMode = (mode: 'agent' | 'user') => {
      if (!metadata?.sessionId) {
        toast.error('Not connected to browser session');
        return;
      }

      console.log(`Switching control mode to: ${mode}`);

      if (mode === 'user') {
        // Call stop to send stopChat action to backend when user takes control
        if (stop) {
          stop();
        }
        setMetadata({
          ...metadata,
          controlMode: mode,
          isFullscreen: !isMobile, // Fullscreen on desktop when user takes control
        });
      } else {
        setMetadata({
          ...metadata,
          controlMode: mode,
          isFullscreen: false,
        });
      }

      toast.success(`Control switched to ${mode} mode`);
    };

    // Global keyboard listener for fullscreen mode (Escape to exit)
    useEffect(() => {
      const handleGlobalKeyDown = (event: KeyboardEvent) => {
        if (event.key === 'Escape' && metadata?.isFullscreen && metadata?.controlMode === 'user') {
          event.preventDefault();
          switchControlMode('agent');
        }
      };

      if (metadata?.isFullscreen && metadata?.controlMode === 'user') {
        document.addEventListener('keydown', handleGlobalKeyDown);
        return () => document.removeEventListener('keydown', handleGlobalKeyDown);
      }
    }, [metadata?.isFullscreen, metadata?.controlMode]);

    if (!metadata) {
      return (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <Loader2 className="size-8 mx-auto mb-2 animate-spin" />
            <p className="text-sm text-muted-foreground">Initializing browser artifact...</p>
          </div>
        </div>
      );
    }

    // Render Browserbase iframe (Live View)
    const renderBrowserbaseIframe = (className?: string) => (
      <iframe
        src={metadata?.liveViewUrl || ''}
        className={className || 'w-full h-full rounded-lg border-0'}
        allow="clipboard-read; clipboard-write"
        sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
        style={{
          // In agent mode, disable pointer events so user can't interfere
          // In user mode, allow full interaction
          pointerEvents: metadata?.controlMode === 'user' ? 'auto' : 'none',
        }}
      />
    );

    // Render loading/error states
    const renderLoadingState = (className?: string) => (
      <div className={`flex items-center justify-center ${className || 'h-full'} bg-gray-50 text-gray-500`}>
        <div className="text-center px-4">
          {metadata.isConnecting ? (
            <>
              <Loader2 className="size-8 mx-auto mb-2 animate-spin" />
              <p className="text-sm">Connecting to browser...</p>
            </>
          ) : (
            <>
              <ClockFading className="size-8 mx-auto mb-2" />
              <p className="text-sm font-medium">Waiting for browser session...</p>
              <p className="text-xs opacity-75">The browser will appear when the agent starts automation</p>
            </>
          )}
        </div>
      </div>
    );

    const renderErrorState = (className?: string) => (
      <div className={`flex items-center justify-center ${className || 'h-full'} bg-gray-50 text-gray-500`}>
        <div className="text-center px-4">
          <MonitorX className="size-8 mx-auto mb-2" />
          <p className="text-sm font-medium">Failed to connect to browser</p>
          <p className="text-xs opacity-75">{metadata.error}</p>
        </div>
      </div>
    );

    // Fullscreen mode when user takes control (desktop only)
    if (metadata.controlMode === 'user' && metadata.isFullscreen) {
      return (
        <div className="fixed inset-0 z-50 browser-fullscreen-bg flex flex-col overflow-hidden">
          {/* Fullscreen header with controls */}
          <div className="sticky top-0 left-0 right-0 z-10 browser-fullscreen-bg">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between px-2 sm:px-4 py-2 sm:py-3 gap-2">
              <div className="flex flex-col gap-1 text-white">
                <div className="flex items-center gap-2">
                  <div className="size-2 bg-red-500 rounded-full animate-pulse status-indicator" />
                  <span className="text-xs sm:text-sm font-medium font-ibm-plex-mono">You're editing manually</span>
                </div>
                <span className="text-xs sm:text-sm text-gray-400 font-inter hidden sm:block">The AI will continue with your changes when you give back control.</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => switchControlMode('agent')}
                className="px-3 sm:px-4 py-2 sm:py-2.5 rounded text-xs sm:text-sm font-medium leading-5 border-0 hover:bg-custom-purple/90 focus:outline-none focus:ring-2 focus:ring-offset-2 bg-custom-purple"
              >
                <div className="flex items-center gap-2 text-white">
                  Give back control
                </div>
              </Button>
            </div>
          </div>

          {/* Fullscreen browser iframe */}
          <div className="flex-1 overflow-hidden browser-fullscreen-bg pt-20 pb-4 sm:pb-12 px-2 sm:px-4 md:px-12">
            {metadata.error ? (
              renderErrorState()
            ) : !metadata.isConnected || !metadata.liveViewUrl ? (
              renderLoadingState()
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <div className="relative rounded-lg shadow-2xl bg-white w-full h-full max-w-[1920px] max-h-[1080px]">
                  {renderBrowserbaseIframe('w-full h-full rounded-lg border-0')}
                </div>
              </div>
            )}
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
              size="lg"
              onClick={() => metadata?.setIsSheetOpen?.(true)}
              className="rounded-full shadow-lg px-4 py-3 bg-custom-purple hover:bg-custom-purple/90 text-white"
            >
              <Monitor className="w-5 h-5 mr-2" />
              View Browser
            </Button>
          </div>

          {/* Mobile: Bottom sheet with browser content */}
          <div className="pointer-events-auto">
            <Sheet open={metadata?.isSheetOpen || false} onOpenChange={metadata?.setIsSheetOpen || (() => {})}>
              <SheetContent side="bottom" className="h-[85vh] p-0 overflow-y-scroll flex flex-col z-[100]">
                <SheetHeader className="px-4 py-3 border-b">
                  <SheetTitle className="text-left">Browser View</SheetTitle>
                </SheetHeader>

                {/* Connection status indicator */}
                {metadata.isConnecting && (
                  <div className="flex items-center justify-center py-2 px-2 text-xs bg-muted/30">
                    <Loader2 className="size-4 mr-2 animate-spin flex-shrink-0" />
                    <span className="truncate">Connecting to browser...</span>
                  </div>
                )}

                {/* Control mode indicator */}
                {metadata.isConnected && metadata.liveViewUrl && (
                  <div className="flex items-center justify-between py-2 px-4 bg-muted/20">
                    <AgentStatusIndicator
                      chatStatus={chatStatus}
                      controlMode={metadata.controlMode}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => switchControlMode(metadata.controlMode === 'user' ? 'agent' : 'user')}
                      className="px-3 py-2 rounded text-xs font-medium border-0 hover:bg-custom-purple/90 bg-custom-purple text-white"
                    >
                      {metadata.controlMode === 'user' ? (
                        <div className="flex items-center gap-2 text-white">
                          Give back control
                        </div>
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
                <div className="flex-1 overflow-y-scroll p-4">
                  {metadata.error ? (
                    renderErrorState('min-h-[400px] rounded-lg')
                  ) : !metadata.isConnected || !metadata.liveViewUrl ? (
                    renderLoadingState('min-h-[400px] rounded-lg')
                  ) : (
                    <div className="flex items-center justify-center">
                      <div className="relative w-full max-w-[768px] h-[400px] bg-white rounded-lg shadow-lg">
                        {renderBrowserbaseIframe('w-full h-full rounded-lg border-0')}
                      </div>
                    </div>
                  )}
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      );
    }

    // Desktop mode
    return (
      <div className="h-full flex flex-col">
        {/* Connection status indicator */}
        {metadata.isConnecting && (
          <div className="flex items-center justify-center py-2 text-sm text-muted-foreground bg-muted/30">
            <Loader2 className="size-4 mr-2 animate-spin" />
            Connecting to browser...
          </div>
        )}

        {/* Control mode indicator */}
        {metadata.isConnected && metadata.liveViewUrl && (
          <div className="flex items-center justify-between py-2 bg-muted/20">
            <AgentStatusIndicator
              chatStatus={chatStatus}
              controlMode={metadata.controlMode}
              className="text-sm text-black"
            />
            <Button
              variant={metadata.controlMode === 'user' ? 'default' : 'outline'}
              size="sm"
              onClick={() => switchControlMode('user')}
              className="px-4 py-2.5 rounded text-sm font-medium leading-5 border-0 hover:bg-custom-purple/90 focus:outline-none focus:ring-2 focus:ring-offset-2 bg-custom-purple"
            >
              <div className="flex items-center gap-2 text-white">
                <MousePointerClick className="w-5 h-5" />
                Take control
              </div>
            </Button>
          </div>
        )}

        {/* Main browser display area */}
        <div className="flex-1 relative m-4">
          {metadata.error ? (
            <div className="absolute inset-0">
              {renderErrorState()}
            </div>
          ) : !metadata.isConnected || !metadata.liveViewUrl ? (
            <div className="absolute inset-0">
              {renderLoadingState()}
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="relative w-full h-full">
                {renderBrowserbaseIframe('w-full h-full rounded-lg border-0 bg-white')}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  },

  actions: [],

  toolbar: [],
});
