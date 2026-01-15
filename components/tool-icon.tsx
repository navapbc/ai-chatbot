'use client';

import {
  Brain,
  Camera,
  Database,
  Download,
  Eye,
  FileText,
  Globe,
  Link,
  MousePointer,
  Search,
  X
} from 'lucide-react';

interface ToolIconProps {
  toolName: string;
  size?: number;
  className?: string;
}

// Icon mapping for tool actions
const getToolIcon = (toolName: string) => {
  const cleanToolName = toolName.replace('tool-', '');

  const iconMap: Record<string, React.ComponentType<any>> = {
    // Stagehand browser tools
    'browser_navigate': Globe,
    'browser_act': MousePointer,
    'browser_extract': Download,
    'browser_observe': Eye,
    'browser_screenshot': Camera,
    'browser_get_url': Link,
    'browser_close': X,
    // Database tools
    'search-participants-by-name': Search,
    'get-participant-with-household': Database,
    'updateWorkingMemory': Brain,
  };

  return iconMap[cleanToolName] || FileText;
};

export const ToolIcon = ({ toolName, size = 12, className = "text-gray-500 flex-shrink-0" }: ToolIconProps) => {
  const IconComponent = getToolIcon(toolName);
  return <IconComponent size={size} className={className} />;
};

// Helper function to get tool display name with icon
export const getToolDisplayInfo = (toolName: string, input?: any): { text: string; icon: React.ComponentType<any> } => {
  const toolMappings: Record<string, (input?: any) => string> = {
    // Stagehand browser tools
    'browser_navigate': (input) => input?.url ? `Navigated to ${input.url}` : 'Navigated to page',
    'browser_act': (input) => input?.action ? `${input.action}` : 'Performed action',
    'browser_extract': (input) => input?.instruction ? `Extracted: ${input.instruction}` : 'Extracted data',
    'browser_observe': (input) => input?.instruction ? `Observed: ${input.instruction}` : 'Observed page',
    'browser_screenshot': () => 'Took screenshot',
    'browser_get_url': () => 'Got current URL',
    'browser_close': () => 'Closed browser',
    // Database tools
    'search-participants-by-name': (input) => input?.name ? `Searched for "${input.name}"` : 'Searched participants',
    'get-participant-with-household': () => 'Retrieved participant data',
    'updateWorkingMemory': () => 'Updated working memory',
  };

  const cleanToolName = toolName.replace('tool-', '');
  const mapper = toolMappings[cleanToolName];

  let text: string;
  if (mapper) {
    text = mapper(input);
  } else {
    // Fallback: convert tool name to readable format
    if (cleanToolName.startsWith('browser_')) {
      const action = cleanToolName.replace('browser_', '').replace(/_/g, ' ');
      text = action.charAt(0).toUpperCase() + action.slice(1);
    } else {
      text = cleanToolName.replace(/-/g, ' ').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
  }

  return {
    text,
    icon: getToolIcon(toolName)
  };
};
