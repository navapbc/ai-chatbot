CREATE TABLE IF NOT EXISTS "EveRun" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"chatId" uuid NOT NULL,
	"userId" uuid NOT NULL,
	"eveSessionId" text NOT NULL,
	"continuationToken" text NOT NULL,
	"streamIndex" integer DEFAULT 0 NOT NULL,
	"relayOrigin" text,
	"pendingRequests" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"progress" jsonb,
	"createdAt" timestamp NOT NULL,
	"updatedAt" timestamp NOT NULL,
	CONSTRAINT "EveRun_userId_chatId_unique" UNIQUE("userId","chatId")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "EveRun" ADD CONSTRAINT "EveRun_userId_User_id_fk" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
