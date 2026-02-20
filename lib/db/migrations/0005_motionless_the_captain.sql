CREATE TABLE IF NOT EXISTS "BrowserSession" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"sessionId" varchar(255) NOT NULL,
	"userId" uuid NOT NULL,
	"chatId" uuid NOT NULL,
	"kernelSessionId" varchar(255) NOT NULL,
	"liveViewUrl" text NOT NULL,
	"cdpWsUrl" text NOT NULL,
	"createdAt" timestamp NOT NULL,
	CONSTRAINT "BrowserSession_sessionId_unique" UNIQUE("sessionId")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "BrowserSession" ADD CONSTRAINT "BrowserSession_userId_User_id_fk" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "BrowserSession" ADD CONSTRAINT "BrowserSession_chatId_Chat_id_fk" FOREIGN KEY ("chatId") REFERENCES "public"."Chat"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
