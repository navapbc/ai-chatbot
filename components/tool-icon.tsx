'use client';

import {
  ArrowLeft,
  Brain,
  Camera,
  CheckSquare,
  Clock,
  Code,
  Database,
  Download,
  FileText,
  Globe,
  Keyboard,
  Maximize2,
  MessageCircle,
  MessageSquare,
  Monitor,
  MousePointer,
  Move,
  Network,
  PanelLeft,
  Search,
  Type,
  Upload,
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
    // agent-browser tools (camelCase format)
    'browserNavigate': Globe,
    'browserClick': MousePointer,
    'browserType': Type,
    'browserFill': FileText,
    'browserSelect': CheckSquare,
    'browserScreenshot': Camera,
    'browserSnapshot': Monitor,
    'browserWait': Clock,
    'browserHover': Move,
    'browserScroll': Move,
    'browserPress': Keyboard,
    'browserCheck': CheckSquare,
    'browserUncheck': X,
    'browserBack': ArrowLeft,
    'browserForward': Globe,
    'browserReload': Globe,
    'browserGetText': FileText,
    'browserGetUrl': Globe,
    'browserGetTitle': FileText,
    // Legacy underscore format (browser_*)
    'browser_navigate': Globe,
    'browser_click': MousePointer,
    'browser_type': Type,
    'browser_fill_form': FileText,
    'browser_select_option': CheckSquare,
    'browser_take_screenshot': Camera,
    'browser_snapshot': Monitor,
    'browser_wait_for': Clock,
    'browser_hover': Move,
    'browser_drag': Move,
    'browser_press_key': Keyboard,
    'browser_evaluate': Code,
    'browser_close': X,
    'browser_resize': Maximize2,
    'browser_tabs': PanelLeft,
    'browser_console_messages': MessageSquare,
    'browser_network_requests': Network,
    'browser_handle_dialog': MessageCircle,
    'browser_file_upload': Upload,
    'browser_install': Download,
    'browser_navigate_back': ArrowLeft,
    // Database/Apricot tools
    'search-participants-by-name': Search,
    'searchApricotUsersByName': Search,
    'getUsersFromApricot': Database,
    'getApricotUserById': Database,
    'getApricotRecordById': Database,
    'getFormsFromApricot': FileText,
    'getApricotFormById': FileText,
    'testApricotAuth': Network,
    'get-participant-with-household': Database,
    'updateWorkingMemory': Brain,
  };

  return iconMap[cleanToolName] || FileText; // Default icon
};

export const ToolIcon = ({ toolName, size = 12, className = "text-gray-500 flex-shrink-0" }: ToolIconProps) => {
  const IconComponent = getToolIcon(toolName);
  
  return <IconComponent size={size} className={className} />;
};

// Helper to truncate URLs for display
const truncateUrl = (url: string, maxLength = 40) => {
  if (!url || url.length <= maxLength) return url;
  try {
    const parsed = new URL(url);
    const domain = parsed.hostname.replace('www.', '');
    const path = parsed.pathname;
    if (domain.length + path.length <= maxLength) {
      return domain + path;
    }
    return domain + (path.length > 10 ? path.slice(0, 10) + '...' : path);
  } catch {
    return url.slice(0, maxLength) + '...';
  }
};

// Helper function to get tool display name with icon
export const getToolDisplayInfo = (toolName: string, input?: any): { text: string; icon: React.ComponentType<any> } => {
  const toolMappings: Record<string, (input?: any) => string> = {
    // agent-browser tools (camelCase format)
    'browserNavigate': (input) => input?.url ? `Opening ${truncateUrl(input.url)}` : 'Opening page',
    'browserClick': (input) => input?.ref ? `Clicking ${input.ref}` : 'Clicking element',
    'browserType': (input) => input?.text ? `Typing "${input.text.slice(0, 30)}${input.text.length > 30 ? '...' : ''}"` : 'Typing text',
    'browserFill': (input) => input?.text ? `Filling "${input.text.slice(0, 20)}${input.text.length > 20 ? '...' : ''}"` : 'Filling field',
    'browserSelect': (input) => input?.value ? `Selecting "${input.value}"` : 'Selecting option',
    'browserScreenshot': () => 'Taking screenshot',
    'browserSnapshot': () => 'Reading page',
    'browserWait': (input) => input?.target ? `Waiting for ${input.target}` : 'Waiting',
    'browserHover': (input) => input?.ref ? `Hovering ${input.ref}` : 'Hovering element',
    'browserScroll': (input) => input?.direction ? `Scrolling ${input.direction}` : 'Scrolling',
    'browserPress': (input) => input?.key ? `Pressing ${input.key}` : 'Pressing key',
    'browserCheck': (input) => input?.ref ? `Checking ${input.ref}` : 'Checking checkbox',
    'browserUncheck': (input) => input?.ref ? `Unchecking ${input.ref}` : 'Unchecking checkbox',
    'browserBack': () => 'Going back',
    'browserForward': () => 'Going forward',
    'browserReload': () => 'Reloading page',
    'browserGetText': (input) => input?.ref ? `Reading text from ${input.ref}` : 'Reading text',
    'browserGetUrl': () => 'Getting URL',
    'browserGetTitle': () => 'Getting title',
    // Legacy underscore format (browser_*)
    'browser_navigate': (input) => input?.url ? `Opening ${truncateUrl(input.url)}` : 'Opening page',
    'browser_click': (input) => input?.element ? `Clicking ${input.element}` : 'Clicking element',
    'browser_type': (input) => input?.text ? `Typing "${input.text.slice(0, 30)}"` : 'Typing text',
    'browser_fill_form': () => 'Filling form',
    'browser_select_option': (input) => input?.values ? `Selecting "${input.values.join(', ')}"` : 'Selecting option',
    'browser_take_screenshot': () => 'Taking screenshot',
    'browser_snapshot': () => 'Reading page',
    'browser_wait_for': (input) => input?.text ? `Waiting for "${input.text}"` : 'Waiting',
    'browser_hover': (input) => input?.element ? `Hovering ${input.element}` : 'Hovering element',
    'browser_drag': () => 'Dragging',
    'browser_press_key': (input) => input?.key ? `Pressing ${input.key}` : 'Pressing key',
    'browser_evaluate': () => 'Running script',
    'browser_close': () => 'Closing browser',
    'browser_resize': () => 'Resizing window',
    'browser_tabs': () => 'Managing tabs',
    'browser_console_messages': () => 'Reading console',
    'browser_network_requests': () => 'Checking network',
    'browser_handle_dialog': () => 'Handling dialog',
    'browser_file_upload': () => 'Uploading file',
    'browser_install': () => 'Installing browser',
    'browser_navigate_back': () => 'Going back',
    // Database/Apricot tools
    'search-participants-by-name': (input) => input?.name ? `Searching "${input.name}"` : 'Searching participants',
    'searchApricotUsersByName': (input) => input?.name ? `Searching "${input.name}"` : 'Searching users',
    'getUsersFromApricot': () => 'Loading users',
    'getApricotUserById': (input) => input?.userId ? `Loading user ${input.userId}` : 'Loading user',
    'getApricotRecordById': (input) => input?.recordId ? `Loading record ${input.recordId}` : 'Loading record',
    'getFormsFromApricot': () => 'Loading forms',
    'getApricotFormById': (input) => input?.formId ? `Loading form ${input.formId}` : 'Loading form',
    'testApricotAuth': () => 'Testing auth',
    'get-participant-with-household': () => 'Loading participant data',
    'updateWorkingMemory': () => 'Updated memory',
  };

  const cleanToolName = toolName.replace('tool-', '');
  const mapper = toolMappings[cleanToolName];

  let text: string;
  if (mapper) {
    text = mapper(input);
  } else {
    // Fallback: convert camelCase or kebab-case to readable format
    text = cleanToolName
      .replace(/([A-Z])/g, ' $1') // camelCase
      .replace(/-/g, ' ') // kebab-case
      .replace(/\b\w/g, l => l.toUpperCase())
      .trim();
  }

  return {
    text,
    icon: getToolIcon(toolName)
  };
};
