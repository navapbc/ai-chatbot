'use client';

import { FlaskConical } from 'lucide-react';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  FEATURE_FLAGS,
  setFlagOverride,
  type FeatureFlagKey,
} from '@/lib/feature-flags';
import { useFeatureFlag } from '@/hooks/use-feature-flag';
import { isProductionEnvironment } from '@/lib/constants';

function FlagRow({ flagKey }: { flagKey: FeatureFlagKey }) {
  const def = FEATURE_FLAGS[flagKey];
  const enabled = useFeatureFlag(flagKey);
  const id = `ff-${flagKey}`;

  return (
    <div className="flex items-start justify-between gap-3 px-2 py-1.5">
      <div className="flex flex-col gap-0.5">
        <Label htmlFor={id} className="text-xs font-medium">
          {def.label}
        </Label>
        <span className="text-[10px] leading-snug text-muted-foreground">
          {def.description}
        </span>
      </div>
      <Switch
        id={id}
        checked={enabled}
        onCheckedChange={(checked) => setFlagOverride(flagKey, checked)}
      />
    </div>
  );
}

// Dev-only menu for toggling feature flags. QA can flip features on/off in
// dev/preview without a redeploy; overrides persist via localStorage. Hidden
// in production.
export function FeatureFlagsMenu() {
  if (isProductionEnvironment) return null;

  const keys = Object.keys(FEATURE_FLAGS) as FeatureFlagKey[];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="flex items-center gap-1.5 h-fit px-2 py-1.5 text-xs text-muted-foreground rounded-md hover:bg-accent hover:text-foreground transition-colors"
        >
          <FlaskConical className="size-3" />
          <span>Flags</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        <DropdownMenuLabel>Feature flags (dev)</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {keys.map((key) => (
          <FlagRow key={key} flagKey={key} />
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
