"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { ChevronDownIcon, CheckIcon } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";
import { Spinner } from "@/components/ui/spinner";

export type TaskItemFileProps = ComponentProps<"div">;

export const TaskItemFile = ({
  children,
  className,
  ...props
}: TaskItemFileProps) => (
  <div
    className={cn(
      "inline-flex items-center gap-1 rounded-md border bg-secondary px-1.5 py-0.5 text-foreground text-xs",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export type TaskItemProps = ComponentProps<"div"> & {
  icon?: ReactNode;
};

export const TaskItem = ({ children, className, icon, ...props }: TaskItemProps) => (
  <div className={cn("text-muted-foreground text-sm flex items-start gap-2", className)} {...props}>
    {icon && <span className="text-muted-foreground shrink-0 mt-0.5">{icon}</span>}
    <span>{children}</span>
  </div>
);

export type TaskProps = ComponentProps<typeof Collapsible>;

export const Task = ({
  defaultOpen = true,
  className,
  ...props
}: TaskProps) => (
  <Collapsible
    className={cn("bg-[#F8F8F8] dark:bg-muted/50 rounded-lg p-3", className)}
    defaultOpen={defaultOpen}
    {...props}
  />
);

export type TaskTriggerProps = ComponentProps<typeof CollapsibleTrigger> & {
  title: string;
  isLoading?: boolean;
  isComplete?: boolean;
  icon?: ReactNode;
};

export const TaskTrigger = ({
  children,
  className,
  title,
  isLoading = false,
  isComplete = false,
  icon,
  ...props
}: TaskTriggerProps) => (
  <CollapsibleTrigger asChild className={cn("group", className)} {...props}>
    {children ?? (
      <div className="flex w-full cursor-pointer items-center gap-2 text-muted-foreground text-sm transition-colors hover:text-foreground">
        {isLoading ? (
          <Spinner className="size-4 text-muted-foreground" />
        ) : isComplete ? (
          <CheckIcon className="size-4 ext-muted-foreground" />
        ) : icon ? (
          <span className="size-4 flex items-center justify-center">{icon}</span>
        ) : null}
        <p className="text-sm flex-1">{title}</p>
        <ChevronDownIcon className="size-4 transition-transform group-data-[state=open]:rotate-180" />
      </div>
    )}
  </CollapsibleTrigger>
);

export type TaskContentProps = ComponentProps<typeof CollapsibleContent>;

export const TaskContent = ({
  children,
  className,
  ...props
}: TaskContentProps) => (
  <CollapsibleContent
    className={cn(
      "data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-top-2 data-[state=open]:slide-in-from-top-2 text-popover-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in",
      className
    )}
    {...props}
  >
    <div className="mt-2 space-y-1.5 border-muted border-l pl-4 ml-2">
      {children}
    </div>
  </CollapsibleContent>
);
