CREATE TABLE IF NOT EXISTS "SessionMapping" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"chatId" uuid NOT NULL,
	"userId" uuid NOT NULL,
	"posthogSessionId" text,
	"kernelSessionId" text,
	"kernelReplayId" text,
	"createdAt" timestamp NOT NULL,
	"updatedAt" timestamp NOT NULL,
	CONSTRAINT "SessionMapping_chatId_unique" UNIQUE("chatId")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "SessionMapping" ADD CONSTRAINT "SessionMapping_chatId_Chat_id_fk" FOREIGN KEY ("chatId") REFERENCES "public"."Chat"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "SessionMapping" ADD CONSTRAINT "SessionMapping_userId_User_id_fk" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
