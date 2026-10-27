import { type InferSelectModel, sql } from 'drizzle-orm';
import {
  pgTable,
  varchar,
  timestamp,
  json,
  uuid,
  text,
  primaryKey,
  foreignKey,
  boolean,
  unique,
  integer,
  jsonb,
  uniqueIndex,
} from 'drizzle-orm/pg-core';
import type { PendingInputRequest } from '../ai/eve/input-requests';
import type { RunProgress } from '../relay/progress';
import type { Participant as ParticipantRecord } from '../data/participants';

export const user = pgTable('User', {
  id: uuid('id').primaryKey().notNull().defaultRandom(),
  email: varchar('email', { length: 64 }).notNull(),
  password: varchar('password', { length: 64 }),
  name: varchar('name', { length: 255 }),
  image: text('image'),
  emailVerified: timestamp('emailVerified'),
});

export type User = InferSelectModel<typeof user>;

export const chat = pgTable('Chat', {
  id: uuid('id').primaryKey().notNull().defaultRandom(),
  createdAt: timestamp('createdAt').notNull(),
  title: text('title').notNull(),
  userId: uuid('userId')
    .notNull()
    .references(() => user.id),
  visibility: varchar('visibility', { enum: ['public', 'private'] })
    .notNull()
    .default('private'),
  mastraThreadId: uuid('mastraThreadId'),
});

export type Chat = InferSelectModel<typeof chat>;

// DEPRECATED: The following schema is deprecated and will be removed in the future.
// Read the migration guide at https://chat-sdk.dev/docs/migration-guides/message-parts
export const messageDeprecated = pgTable('Message', {
  id: uuid('id').primaryKey().notNull().defaultRandom(),
  chatId: uuid('chatId')
    .notNull()
    .references(() => chat.id),
  role: varchar('role').notNull(),
  content: json('content').notNull(),
  createdAt: timestamp('createdAt').notNull(),
});

export type MessageDeprecated = InferSelectModel<typeof messageDeprecated>;

export const message = pgTable('Message_v2', {
  id: uuid('id').primaryKey().notNull().defaultRandom(),
  chatId: uuid('chatId')
    .notNull()
    .references(() => chat.id),
  role: varchar('role').notNull(),
  parts: json('parts').notNull(),
  attachments: json('attachments').notNull(),
  createdAt: timestamp('createdAt').notNull(),
});

export type DBMessage = InferSelectModel<typeof message>;

// DEPRECATED: The following schema is deprecated and will be removed in the future.
// Read the migration guide at https://chat-sdk.dev/docs/migration-guides/message-parts
export const voteDeprecated = pgTable(
  'Vote',
  {
    chatId: uuid('chatId')
      .notNull()
      .references(() => chat.id),
    messageId: uuid('messageId')
      .notNull()
      .references(() => messageDeprecated.id),
    isUpvoted: boolean('isUpvoted').notNull(),
  },
  (table) => {
    return {
      pk: primaryKey({ columns: [table.chatId, table.messageId] }),
    };
  },
);

export type VoteDeprecated = InferSelectModel<typeof voteDeprecated>;

export const vote = pgTable(
  'Vote_v2',
  {
    chatId: uuid('chatId')
      .notNull()
      .references(() => chat.id),
    messageId: uuid('messageId')
      .notNull()
      .references(() => message.id),
    isUpvoted: boolean('isUpvoted').notNull(),
  },
  (table) => {
    return {
      pk: primaryKey({ columns: [table.chatId, table.messageId] }),
    };
  },
);

export type Vote = InferSelectModel<typeof vote>;

export const document = pgTable(
  'Document',
  {
    id: uuid('id').notNull().defaultRandom(),
    createdAt: timestamp('createdAt').notNull(),
    title: text('title').notNull(),
    content: text('content'),
    kind: varchar('text', {
      enum: ['text', 'code', 'image', 'sheet', 'browser'],
    })
      .notNull()
      .default('text'),
    userId: uuid('userId')
      .notNull()
      .references(() => user.id),
  },
  (table) => {
    return {
      pk: primaryKey({ columns: [table.id, table.createdAt] }),
    };
  },
);

export type Document = InferSelectModel<typeof document>;

export const suggestion = pgTable(
  'Suggestion',
  {
    id: uuid('id').notNull().defaultRandom(),
    documentId: uuid('documentId').notNull(),
    documentCreatedAt: timestamp('documentCreatedAt').notNull(),
    originalText: text('originalText').notNull(),
    suggestedText: text('suggestedText').notNull(),
    description: text('description'),
    isResolved: boolean('isResolved').notNull().default(false),
    userId: uuid('userId')
      .notNull()
      .references(() => user.id),
    createdAt: timestamp('createdAt').notNull(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.id] }),
    documentRef: foreignKey({
      columns: [table.documentId, table.documentCreatedAt],
      foreignColumns: [document.id, document.createdAt],
    }),
  }),
);

export type Suggestion = InferSelectModel<typeof suggestion>;

export const stream = pgTable(
  'Stream',
  {
    id: uuid('id').notNull().defaultRandom(),
    chatId: uuid('chatId').notNull(),
    createdAt: timestamp('createdAt').notNull(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.id] }),
    chatRef: foreignKey({
      columns: [table.chatId],
      foreignColumns: [chat.id],
    }),
  }),
);

export type Stream = InferSelectModel<typeof stream>;

// Mapping table correlating a chat with its PostHog session and Kernel browser
// session/replay. Populated from two directions: the client supplies the
// PostHog session id (only knowable in the browser), the server supplies the
// Kernel session/replay ids (only knowable when the browser is created). One
// row per chat — fields fill in as each id materializes (upsert on chatId).
export const sessionMapping = pgTable(
  'SessionMapping',
  {
    id: uuid('id').primaryKey().notNull().defaultRandom(),
    chatId: uuid('chatId')
      .notNull()
      .references(() => chat.id),
    userId: uuid('userId')
      .notNull()
      .references(() => user.id),
    posthogSessionId: text('posthogSessionId'),
    // Deep-link to the PostHog-hosted session replay. PostHog has no video
    // export, so we keep the link rather than re-storing the recording.
    posthogReplayUrl: text('posthogReplayUrl'),
    kernelSessionId: text('kernelSessionId'),
    kernelReplayId: text('kernelReplayId'),
    // Stored location of the kernel browser video, downloaded from Kernel and
    // uploaded to object storage on browser teardown. Null until the video
    // lands. Storage-agnostic (currently a GCS URL).
    kernelReplayUrl: text('kernelReplayUrl'),
    createdAt: timestamp('createdAt').notNull(),
    updatedAt: timestamp('updatedAt').notNull(),
  },
  (table) => ({
    chatUnique: unique('SessionMapping_chatId_unique').on(table.chatId),
  }),
);

export type SessionMapping = InferSelectModel<typeof sessionMapping>;

// One row per Eve-driven chat run, replacing the in-memory continuity map.
// chatId deliberately has no foreign key: the eve-chat route does not create
// Chat rows. The continuation token is an Eve session handle; no relay token
// or credential is ever stored here.
export const eveRun = pgTable(
  'EveRun',
  {
    id: uuid('id').primaryKey().notNull().defaultRandom(),
    chatId: uuid('chatId').notNull(),
    userId: uuid('userId')
      .notNull()
      .references(() => user.id),
    eveSessionId: text('eveSessionId').notNull(),
    continuationToken: text('continuationToken').notNull(),
    // Count of stream events already forwarded; the next read's startIndex.
    streamIndex: integer('streamIndex').notNull().default(0),
    // Exact https origin of the relay page; set only for extension runs.
    relayOrigin: text('relayOrigin'),
    pendingRequests: jsonb('pendingRequests')
      .$type<PendingInputRequest[]>()
      .notNull()
      .default([]),
    progress: jsonb('progress').$type<RunProgress>(),
    createdAt: timestamp('createdAt').notNull(),
    updatedAt: timestamp('updatedAt').notNull(),
  },
  (table) => ({
    userChatUnique: unique('EveRun_userId_chatId_unique').on(
      table.userId,
      table.chatId,
    ),
    // One relay run per chat id across all users; web rows are unaffected.
    relayChatUnique: uniqueIndex('EveRun_chatId_relay_unique')
      .on(table.chatId)
      .where(sql`${table.relayOrigin} IS NOT NULL`),
  }),
);

export type EveRun = InferSelectModel<typeof eveRun>;

// Participant records for the extension's participant picker. The full record
// is read only on the server; the API exposes a summary. searchText is
// lowercase "first middle last recordId" and is what prefix search matches.
export const participant = pgTable('Participant', {
  recordId: text('recordId').primaryKey().notNull(),
  displayName: text('displayName').notNull(),
  county: text('county').notNull(),
  searchText: text('searchText').notNull(),
  record: jsonb('record').$type<ParticipantRecord>().notNull(),
  createdAt: timestamp('createdAt').notNull().defaultNow(),
});

export type ParticipantRow = InferSelectModel<typeof participant>;
