CREATE TABLE IF NOT EXISTS "Participant" (
	"recordId" text PRIMARY KEY NOT NULL,
	"displayName" text NOT NULL,
	"county" text NOT NULL,
	"searchText" text NOT NULL,
	"record" jsonb NOT NULL,
	"createdAt" timestamp DEFAULT now() NOT NULL
);
