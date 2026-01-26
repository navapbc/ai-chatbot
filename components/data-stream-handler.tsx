'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useSWRConfig } from 'swr';
import { artifactDefinitions } from './artifact';
import { initialArtifactData, useArtifact } from '@/hooks/use-artifact';
import { useDataStream } from './data-stream-provider';

// Store for real-time browser actions (accessible to message components)
export type BrowserActionEvent = {
  action: string;
  status: 'running' | 'complete' | 'error';
  command?: string;
  timestamp: number;
  duration?: number;
  error?: string;
};

// Global store for current browser action (for real-time UI updates)
let currentBrowserAction: BrowserActionEvent | null = null;
let browserActionListeners: Set<(action: BrowserActionEvent | null) => void> = new Set();

export function getCurrentBrowserAction() {
  return currentBrowserAction;
}

export function subscribeToBrowserAction(listener: (action: BrowserActionEvent | null) => void) {
  browserActionListeners.add(listener);
  // Immediately call with current state
  listener(currentBrowserAction);
  return () => {
    browserActionListeners.delete(listener);
  };
}

function setBrowserAction(action: BrowserActionEvent | null) {
  currentBrowserAction = action;
  browserActionListeners.forEach(listener => listener(action));
}

export function DataStreamHandler() {
  const { dataStream } = useDataStream();
  const { mutate } = useSWRConfig();

  const { artifact, setArtifact, setMetadata } = useArtifact();
  const lastProcessedIndex = useRef(-1);

  // Track artifact kind locally to handle stream parts that arrive before state updates
  const currentKindRef = useRef(artifact.kind);
  // Track documentId locally to enable metadata writes before React state updates
  const currentDocumentIdRef = useRef(artifact.documentId);

  // Keep refs in sync with artifact state
  useEffect(() => {
    currentKindRef.current = artifact.kind;
  }, [artifact.kind]);

  useEffect(() => {
    currentDocumentIdRef.current = artifact.documentId;
  }, [artifact.documentId]);

  // Create a setMetadata function that uses the local ref for documentId
  // This ensures metadata can be written even before React state updates
  const setMetadataWithCurrentId = useCallback(
    (updater: any) => {
      const docId = currentDocumentIdRef.current;
      if (docId && docId !== 'init') {
        mutate(
          `artifact-metadata-${docId}`,
          (current: any) => {
            const result = typeof updater === 'function' ? updater(current || {}) : updater;
            return result;
          },
          { revalidate: false }
        );
      }
    },
    [mutate]
  );

  useEffect(() => {
    if (!dataStream?.length) return;

    const newDeltas = dataStream.slice(lastProcessedIndex.current + 1);
    lastProcessedIndex.current = dataStream.length - 1;

    newDeltas.forEach((delta) => {
      // For data-kind stream parts, update our local tracking
      if (delta.type === 'data-kind' && typeof delta.data === 'string') {
        currentKindRef.current = delta.data;
      }

      // For data-id stream parts, update our local tracking immediately
      // This ensures setMetadataWithCurrentId can write to the correct key
      if (delta.type === 'data-id' && typeof delta.data === 'string') {
        currentDocumentIdRef.current = delta.data;
      }

      // Handle browser action events for real-time status updates
      if (delta.type === 'data-browserAction') {
        const data = delta.data as any;
        if (data?.type === 'browser-action') {
          const actionEvent: BrowserActionEvent = {
            action: data.action,
            status: data.status || 'running',
            command: data.command,
            timestamp: data.timestamp || Date.now(),
            duration: data.duration,
            error: data.error,
          };

          // Set the current action (running) or clear it (complete/error)
          if (actionEvent.status === 'running') {
            setBrowserAction(actionEvent);
          } else {
            // Brief delay before clearing to show completion
            setTimeout(() => {
              if (currentBrowserAction?.timestamp === actionEvent.timestamp) {
                setBrowserAction(null);
              }
            }, 300);
          }
        }
      }

      const artifactDefinition = artifactDefinitions.find(
        (artifactDefinition) => artifactDefinition.kind === currentKindRef.current,
      );

      if (artifactDefinition?.onStreamPart) {
        artifactDefinition.onStreamPart({
          streamPart: delta,
          setArtifact,
          // Use the ref-based setMetadata to ensure metadata writes work
          // even before React state updates complete
          setMetadata: setMetadataWithCurrentId,
        });
      }

      setArtifact((draftArtifact) => {
        if (!draftArtifact) {
          return { ...initialArtifactData, status: 'streaming' };
        }

        switch (delta.type) {
          case 'data-id':
            return {
              ...draftArtifact,
              documentId: delta.data,
              status: 'streaming',
            };

          case 'data-title':
            return {
              ...draftArtifact,
              title: delta.data,
              status: 'streaming',
            };

          case 'data-kind':
            return {
              ...draftArtifact,
              kind: delta.data,
              status: 'streaming',
            };

          case 'data-clear':
            return {
              ...draftArtifact,
              content: '',
              status: 'streaming',
            };

          case 'data-finish':
            return {
              ...draftArtifact,
              status: 'idle',
            };

          default:
            return draftArtifact;
        }
      });
    });
  }, [dataStream, setArtifact, setMetadata, artifact]);

  return null;
}
