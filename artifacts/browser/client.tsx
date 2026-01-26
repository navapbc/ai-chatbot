import { Artifact } from '@/components/create-artifact';
import { Button } from '@/components/ui/button';
import { Monitor, MousePointerClick } from 'lucide-react';
import { toast } from 'sonner';
import { useIsMobile } from '@/hooks/use-mobile';
import { memo, useRef, useEffect } from 'react';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { AgentStatusIndicator } from '@/components/agent-status-indicator';
import { BrowserLoadingState } from './browser-states';

// Memoized iframe component to prevent WebRTC reconnection on parent re-renders
const KernelLiveViewIframe = memo(function KernelLiveViewIframe({
  src,
  pointerEvents,
  className,
  style,
}: {
  src: string;
  pointerEvents: 'auto' | 'none';
  className?: string;
  style?: React.CSSProperties;
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Only update pointer-events style without remounting iframe
  useEffect(() => {
    if (iframeRef.current) {
      iframeRef.current.style.pointerEvents = pointerEvents;
    }
  }, [pointerEvents]);

  return (
    <iframe
      ref={iframeRef}
      src={src}
      className={className}
      style={{ ...style, pointerEvents }}
      allow="autoplay; clipboard-read; clipboard-write; fullscreen"
      title="Browser Live View"
    />
  );
}, (prevProps, nextProps) => {
  // Only re-render if src changes (which means a new session)
  // Ignore pointerEvents changes - we handle those via useEffect
  return prevProps.src === nextProps.src;
});

interface BrowserArtifactMetadata {
  sessionId: string;
  liveViewUrl: string;
  controlMode: 'agent' | 'user';
  isFullscreen: boolean;
  isSheetOpen?: boolean;
  setIsSheetOpen?: (open: boolean) => void;
}

export const browserArtifact = new Artifact<'browser', BrowserArtifactMetadata>({
  kind: 'browser',
  description: 'Live browser automation display using Kernel Live View',

  initialize: async ({ setMetadata }) => {
    setMetadata({
      sessionId: '',
      liveViewUrl: '',
      controlMode: 'agent',
      isFullscreen: false,
    });
  },

  onStreamPart: ({ streamPart, setMetadata, setArtifact }) => {
    // Handle artifact creation - make it visible when browser kind is signaled
    if (streamPart.type === 'data-kind' && streamPart.data === 'browser') {
      setArtifact((draftArtifact) => ({
        ...draftArtifact,
        isVisible: true,
        status: 'streaming',
      }));
    }

    // Handle browser session info from the stream
    if (streamPart.type === 'data-browserSession') {
      const data = streamPart.data as any;
      if (data?.type === 'browser-session') {
        // Ensure artifact is visible
        setArtifact((draftArtifact) => ({
          ...draftArtifact,
          isVisible: true,
          status: 'streaming',
        }));
        // Update metadata with session info
        setMetadata((prev) => ({
          ...prev,
          sessionId: data.sessionId,
          liveViewUrl: data.liveViewUrl,
        }));
      }
    }

    // When we see a browser action starting (like browserNavigate), ensure artifact is visible
    if (streamPart.type === 'data-browserAction') {
      const data = streamPart.data as any;
      if (data?.type === 'browser-action' && data?.status === 'running') {
        setArtifact((draftArtifact) => ({
          ...draftArtifact,
          isVisible: true,
          status: 'streaming',
        }));
      }
    }
  },

  content: ({ metadata, setMetadata, chatStatus, stop }) => {
    const isMobile = useIsMobile();

    const switchControlMode = (mode: 'agent' | 'user') => {
      if (mode === 'user' && stop) {
        stop();
      }
      setMetadata((prev) => ({
        ...prev,
        controlMode: mode,
        isFullscreen: mode === 'user' && !isMobile,
      }));
      toast.success(`Control switched to ${mode} mode`);
    };

    if (!metadata?.liveViewUrl) {
      return <BrowserLoadingState />;
    }

    // Use CSS pointer-events to control interaction instead of URL params
    // This keeps the WebRTC connection alive when switching modes
    const isUserControl = metadata.controlMode === 'user';
    const isFullscreen = isUserControl && metadata.isFullscreen && !isMobile;

    // Desktop: Use CSS-based fullscreen approach to avoid iframe remounting
    // The iframe stays in the same DOM position, we just change the container styling
    if (!isMobile) {
      return (
        <div className={isFullscreen
          ? "fixed inset-0 z-50 browser-fullscreen-bg flex flex-col overflow-hidden"
          : "h-full flex flex-col"
        }>
          {/* Header with controls - different styling for fullscreen vs normal */}
          <div className={isFullscreen
            ? "sticky top-0 left-0 right-0 z-10 browser-fullscreen-bg"
            : "flex items-center justify-between py-2 bg-muted/20"
          }>
            {isFullscreen ? (
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between px-2 sm:px-4 py-2 sm:py-3 gap-2">
                <div className="flex flex-col gap-1 text-white">
                  <div className="flex items-center gap-2">
                    <div className="size-2 bg-red-500 rounded-full animate-pulse status-indicator" />
                    <span className="text-xs sm:text-sm font-medium font-ibm-plex-mono">
                      You're editing manually
                    </span>
                  </div>
                  <span className="text-xs sm:text-sm text-gray-400 font-inter hidden sm:block">
                    The AI will continue with your changes when you give back
                    control.
                  </span>
                </div>
                <div className="flex items-center gap-2">
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
            ) : (
              <>
                <AgentStatusIndicator
                  chatStatus={chatStatus}
                  controlMode={metadata.controlMode}
                  className="text-sm text-black"
                />
                <div className="flex items-center gap-2">
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
              </>
            )}
          </div>

          {/* Browser iframe container - single instance, styling changes based on mode */}
          <div className={isFullscreen
            ? "flex-1 overflow-hidden browser-fullscreen-bg pt-20 pb-4 sm:pb-12 px-2 sm:px-4 md:px-12"
            : "flex-1 relative m-4 flex items-center justify-center"
          }>
            <div className="w-full h-full flex items-center justify-center">
              <KernelLiveViewIframe
                src={metadata.liveViewUrl}
                pointerEvents={isUserControl ? 'auto' : 'none'}
                className={isFullscreen
                  ? "aspect-video border-0 rounded-lg bg-white max-w-full max-h-full"
                  : "w-full aspect-video border-0 rounded-lg bg-white max-h-full"
                }
                style={isFullscreen ? { width: 'min(100%, calc((100vh - 120px) * 16 / 9))' } : undefined}
              />
            </div>
          </div>
        </div>
      );
    }

    // Mobile drawer mode - separate since Sheet component handles its own mounting
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
          <Sheet
            open={metadata?.isSheetOpen || false}
            onOpenChange={metadata?.setIsSheetOpen || (() => {})}
          >
            <SheetContent
              side="bottom"
              className="h-[85vh] p-0 overflow-y-scroll flex flex-col z-[100]"
            >
              <SheetHeader className="px-4 py-3 border-b">
                <SheetTitle className="text-left">Browser View</SheetTitle>
              </SheetHeader>

              {/* Control mode indicator */}
              <div className="flex items-center justify-between py-2 px-4 bg-muted/20">
                <AgentStatusIndicator
                  chatStatus={chatStatus}
                  controlMode={metadata.controlMode}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    switchControlMode(
                      metadata.controlMode === 'user' ? 'agent' : 'user'
                    )
                  }
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

              {/* Browser iframe container */}
              <div className="flex-1 overflow-hidden p-4 flex items-center justify-center">
                <KernelLiveViewIframe
                  src={metadata.liveViewUrl}
                  pointerEvents={isUserControl ? 'auto' : 'none'}
                  className="w-full aspect-video border-0 rounded-lg bg-white max-h-full"
                />
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    );
  },

  actions: [],

  toolbar: [],
});
