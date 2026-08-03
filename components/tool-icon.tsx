'use client';

import {
  ArrowLeft,
  Camera,
  CheckSquare,
  Clock,
  Code,
  Database,
  Download,
  FileText,
  Globe,
  Keyboard,
  Layers,
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
    // New toolset format (browser_*)
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
    // Legacy format (playwright_browser_*)
    'playwright_browser_navigate': Globe,
    'playwright_browser_click': MousePointer,
    'playwright_browser_type': Type,
    'playwright_browser_fill_form': FileText,
    'playwright_browser_select_option': CheckSquare,
    'playwright_browser_take_screenshot': Camera,
    'playwright_browser_snapshot': Monitor,
    'playwright_browser_wait_for': Clock,
    'playwright_browser_hover': Move,
    'playwright_browser_drag': Move,
    'playwright_browser_press_key': Keyboard,
    'playwright_browser_evaluate': Code,
    'playwright_browser_close': X,
    'playwright_browser_resize': Maximize2,
    'playwright_browser_tabs': PanelLeft,
    'playwright_browser_console_messages': MessageSquare,
    'playwright_browser_network_requests': Network,
    'playwright_browser_handle_dialog': MessageCircle,
    'playwright_browser_file_upload': Upload,
    'playwright_browser_install': Download,
    'playwright_browser_navigate_back': ArrowLeft,
    // Database tools
    'search-participants-by-name': Search,
    'get-participant-with-household': Database,
    'gapAnalysis': FileText,
    'actionLabel': Layers,
    'readReference': FileText,
  };

  return iconMap[cleanToolName] || FileText; // Default icon
};

export const ToolIcon = ({ toolName, size = 12, className = "text-gray-500 flex-shrink-0" }: ToolIconProps) => {
  const IconComponent = getToolIcon(toolName);
  
  return <IconComponent size={size} className={className} />;
};

// Map agent-browser CLI commands to display info
const browserCommandMap: Record<string, { verb: string; icon: React.ComponentType<any> }> = {
  'open': { verb: 'Opening', icon: Globe },
  'goto': { verb: 'Opening', icon: Globe },
  'navigate': { verb: 'Opening', icon: Globe },
  'snapshot': { verb: 'Reading page', icon: Monitor },
  'click': { verb: 'Clicking', icon: MousePointer },
  'dblclick': { verb: 'Double-clicking', icon: MousePointer },
  'fill': { verb: 'Filling', icon: Type },
  'type': { verb: 'Typing', icon: Type },
  'press': { verb: 'Pressing', icon: Keyboard },
  'select': { verb: 'Selecting', icon: CheckSquare },
  'check': { verb: 'Checking', icon: CheckSquare },
  'uncheck': { verb: 'Unchecking', icon: CheckSquare },
  'hover': { verb: 'Hovering', icon: Move },
  'focus': { verb: 'Focusing', icon: MousePointer },
  'scroll': { verb: 'Scrolling', icon: Move },
  'scrollintoview': { verb: 'Scrolling to', icon: Move },
  'wait': { verb: 'Waiting', icon: Clock },
  'get': { verb: 'Getting', icon: Search },
  'screenshot': { verb: 'Taking screenshot', icon: Camera },
  'drag': { verb: 'Dragging', icon: Move },
  'upload': { verb: 'Uploading', icon: Upload },
  'eval': { verb: 'Running script', icon: Code },
  'find': { verb: 'Finding', icon: Search },
  'back': { verb: 'Going back', icon: ArrowLeft },
  'forward': { verb: 'Going forward', icon: Globe },
  'reload': { verb: 'Reloading', icon: Globe },
  'close': { verb: 'Closing', icon: X },
};

/** Truncate a user-visible value for the one-line tool label. */
const truncate = (value: string, max: number): string =>
  value.length > max ? `${value.substring(0, max)}...` : value;

/** Positional arguments of an agent-browser command, minus its flags. */
const positionalArgs = (argv: string[]): string[] => {
  const out: string[] = [];
  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith('-')) {
      // Flags that take a value consume the next token (e.g. `-s form`).
      if (['-s', '--selector', '-d', '--depth', '--name', '--load', '--url', '--text', '--fn'].includes(arg)) i++;
      continue;
    }
    out.push(arg);
  }
  return out;
};

// Parse an agent-browser argv command into display text and icon.
const parseBrowserAction = (input?: Record<string, any>): { text: string; icon: React.ComponentType<any> } => {
  const argv: string[] = Array.isArray(input?.command) ? input.command.map(String) : [];
  if (argv.length === 0) return { text: 'Browser', icon: Monitor };

  const command = argv[0].toLowerCase();
  const mapping = browserCommandMap[command];
  if (!mapping) return { text: `Browser: ${command}`, icon: Monitor };

  const args = positionalArgs(argv);

  switch (command) {
    // Commands whose arguments add nothing a caseworker would read.
    case 'snapshot':
    case 'screenshot':
    case 'back':
    case 'forward':
    case 'reload':
    case 'close':
      return { text: mapping.verb, icon: mapping.icon };

    case 'open':
    case 'goto':
    case 'navigate':
      return args[0]
        ? { text: `${mapping.verb} ${truncate(args[0], 40)}`, icon: mapping.icon }
        : { text: mapping.verb, icon: mapping.icon };

    // `fill <sel> <text>`, `type <sel> <text>`, `select <sel> <value...>` —
    // the value the caseworker cares about is everything after the selector.
    case 'fill':
    case 'type':
    case 'select': {
      const value = args.slice(1).join(', ');
      return value
        ? { text: `${mapping.verb} "${truncate(value, 35)}"`, icon: mapping.icon }
        : { text: mapping.verb, icon: mapping.icon };
    }

    case 'press':
      return args[0]
        ? { text: `${mapping.verb} ${args[0]}`, icon: mapping.icon }
        : { text: mapping.verb, icon: mapping.icon };

    case 'wait': {
      const [target] = args;
      if (!target) return { text: mapping.verb, icon: mapping.icon };
      return Number.isFinite(Number(target))
        ? { text: `${mapping.verb} ${target}ms`, icon: mapping.icon }
        : { text: `${mapping.verb} for ${target}`, icon: mapping.icon };
    }

    // `find <locator> <value> [action]` — e.g. find label "Email" fill
    case 'find': {
      const [, value, action] = args;
      if (!value) return { text: mapping.verb, icon: mapping.icon };
      const verb = action === 'fill' ? 'Filling' : action === 'click' ? 'Clicking' : 'Using';
      return {
        text: `${verb} "${truncate(value, 30)}"`,
        icon: action === 'fill' ? Type : MousePointer,
      };
    }

    default:
      return { text: mapping.verb, icon: mapping.icon };
  }
};

const hasValue = (v: any): boolean =>
  Array.isArray(v) ? v.length > 0 : v != null && String(v).length > 0;

// A tool call is "specific" when it carries a concrete user-visible value
// (a typed value, a chosen option, a destination URL) rather than a generic
// navigation/inspection action (click, snapshot, scroll, wait, ...).
export const isSpecificToolAction = (toolName: string, input?: any): boolean => {
  const cleanToolName = toolName.replace('tool-', '');

  if (cleanToolName === 'browser') {
    const argv: string[] = Array.isArray(input?.command) ? input.command.map(String) : [];
    if (argv.length === 0) return false;
    const args = positionalArgs(argv);
    switch (argv[0].toLowerCase()) {
      // The concrete value follows the selector.
      case 'fill':
      case 'type':
      case 'select':
        return hasValue(args.slice(1).join(''));
      case 'open':
      case 'goto':
      case 'navigate':
        return hasValue(args[0]);
      // find <locator> <value> fill — only the fill variant carries a value.
      case 'find':
        return args[2] === 'fill' && hasValue(args[3]);
      default:
        return false;
    }
  }

  // Legacy browser_* / playwright_browser_* tool formats
  const base = cleanToolName.replace(/^playwright_/, '');
  switch (base) {
    case 'browser_type': return hasValue(input?.text);
    case 'browser_select_option': return hasValue(input?.values);
    case 'browser_navigate': return hasValue(input?.url);
    default: return false;
  }
};

// Helper function to get tool display name with icon
export const getToolDisplayInfo = (toolName: string, input?: any): { text: string; icon: React.ComponentType<any> } => {
  // Handle AI SDK browser tool (agent-browser)
  const cleanToolName = toolName.replace('tool-', '');
  if (cleanToolName === 'browser') {
    return parseBrowserAction(input);
  }

  const toolMappings: Record<string, (input?: any) => string> = {
    // New toolset format (browser_*)
    'browser_navigate': (input) => input?.url ? `Navigated to ${input.url}` : 'Navigated to page',
    'browser_click': (input) => input?.element ? `Clicked on ${input.element}` : 'Clicked element',
    'browser_type': (input) => input?.text ? `Typed "${input.text}"` : 'Typed text',
    'browser_fill_form': () => 'Filled form fields',
    'browser_select_option': (input) => input?.values ? `Selected "${input.values.join(', ')}"` : 'Selected option',
    'browser_take_screenshot': () => 'Took screenshot',
    'browser_snapshot': () => 'Captured page snapshot',
    'browser_wait_for': (input) => input?.text ? `Waited for "${input.text}"` : 'Waited for element',
    'browser_hover': (input) => input?.element ? `Hovered over ${input.element}` : 'Hovered over element',
    'browser_drag': () => 'Performed drag and drop',
    'browser_press_key': (input) => input?.key ? `Pressed key "${input.key}"` : 'Pressed key',
    'browser_evaluate': () => 'Executed JavaScript',
    'browser_close': () => 'Closed browser',
    'browser_resize': () => 'Resized browser window',
    'browser_tabs': () => 'Managed browser tabs',
    'browser_console_messages': () => 'Retrieved console messages',
    'browser_network_requests': () => 'Retrieved network requests',
    'browser_handle_dialog': () => 'Handled dialog',
    'browser_file_upload': () => 'Uploaded files',
    'browser_install': () => 'Installed browser',
    'browser_navigate_back': () => 'Navigated back',
    // Legacy format (playwright_browser_*)
    'playwright_browser_navigate': (input) => input?.url ? `Navigated to ${input.url}` : 'Navigated to page',
    'playwright_browser_click': (input) => input?.element ? `Clicked on ${input.element}` : 'Clicked element',
    'playwright_browser_type': (input) => input?.text ? `Typed "${input.text}"` : 'Typed text',
    'playwright_browser_fill_form': () => 'Filled form fields',
    'playwright_browser_select_option': (input) => input?.values ? `Selected "${input.values.join(', ')}"` : 'Selected option',
    'playwright_browser_take_screenshot': () => 'Took screenshot',
    'playwright_browser_snapshot': () => 'Captured page snapshot',
    'playwright_browser_wait_for': (input) => input?.text ? `Waited for "${input.text}"` : 'Waited for element',
    'playwright_browser_hover': (input) => input?.element ? `Hovered over ${input.element}` : 'Hovered over element',
    'playwright_browser_drag': () => 'Performed drag and drop',
    'playwright_browser_press_key': (input) => input?.key ? `Pressed key "${input.key}"` : 'Pressed key',
    'playwright_browser_evaluate': () => 'Executed JavaScript',
    'playwright_browser_close': () => 'Closed browser',
    'playwright_browser_resize': () => 'Resized browser window',
    'playwright_browser_tabs': () => 'Managed browser tabs',
    'playwright_browser_console_messages': () => 'Retrieved console messages',
    'playwright_browser_network_requests': () => 'Retrieved network requests',
    'playwright_browser_handle_dialog': () => 'Handled dialog',
    'playwright_browser_file_upload': () => 'Uploaded files',
    'playwright_browser_install': () => 'Installed browser',
    'playwright_browser_navigate_back': () => 'Navigated back',
    // Database tools
    'search-participants-by-name': (input) => input?.name ? `Searched for participant "${input.name}"` : 'Searched for participant',
    'get-participant-with-household': () => 'Retrieved participant data',
    'gapAnalysis': () => 'Gap analysis',
    'readReference': (input) => input?.path ? `Loaded ${input.path}` : 'Loaded reference file',
  };

  const mapper = toolMappings[cleanToolName];
  
  let text: string;
  if (mapper) {
    text = mapper(input);
  } else {
    // Fallback: convert kebab-case to readable format
    text = cleanToolName.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
  
  return {
    text,
    icon: getToolIcon(toolName)
  };
};
