import 'server-only';

import {
  and,
  asc,
  count,
  desc,
  eq,
  gt,
  gte,
  inArray,
  lt,
  type SQL,
} from 'drizzle-orm';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';

import {
  user,
  chat,
  type User,
  document,
  type Suggestion,
  suggestion,
  message,
  vote,
  type DBMessage,
  type Chat,
  stream,
  sessionMapping,
  eveRun,
  type EveRun,
} from './schema';
import type { ArtifactKind } from '@/components/artifact';
import { generateUUID } from '../utils';
import { generateHashedPassword } from './utils';
import type { VisibilityType } from '@/components/visibility-selector';
import { ChatSDKError } from '../errors';
import type { PendingInputRequest } from '../ai/eve/input-requests';
import type { RunProgress } from '../relay/progress';

// Optionally, if not using email/pass login, you can
// use the Drizzle adapter for Auth.js / NextAuth
// https://authjs.dev/reference/adapter/drizzle

// biome-ignore lint: Forbidden non-null assertion.
const client = postgres(process.env.DATABASE_URL!);
const db = drizzle(client);

export async function getUser(email: string): Promise<Array<User>> {
  try {
    return await db.select().from(user).where(eq(user.email, email));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get user by email',
    );
  }
}

export async function getUserById({
  id,
}: { id: string }): Promise<User | null> {
  try {
    const result = await db.select().from(user).where(eq(user.id, id));
    return result[0] || null;
  } catch (error) {
    return null;
  }
}

export async function ensureUserExists({
  id,
  email,
}: { id: string; email: string }): Promise<User> {
  const existingUser = await getUserById({ id });
  if (existingUser) {
    return existingUser;
  }

  const password = generateHashedPassword(generateUUID());

  try {
    const result = await db
      .insert(user)
      .values({
        id,
        email,
        password,
      })
      .returning();
    return result[0];
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to create user');
  }
}

export async function createUser(email: string, password: string) {
  const hashedPassword = generateHashedPassword(password);

  try {
    return await db.insert(user).values({ email, password: hashedPassword });
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to create user');
  }
}

export async function upsertOAuthUser({
  email,
  name,
  image,
}: {
  email: string;
  name?: string | null;
  image?: string | null;
}) {
  try {
    const [existingUser] = await getUser(email);

    if (existingUser) {
      // Update existing user profile with OAuth data
      const [updatedUser] = await db
        .update(user)
        .set({
          name: name ?? existingUser.name,
          image: image ?? existingUser.image,
          emailVerified: new Date(),
        })
        .where(eq(user.email, email))
        .returning();
      return updatedUser;
    }

    // Create new OAuth user (no password)
    const [newUser] = await db
      .insert(user)
      .values({
        email,
        name,
        image,
        password: null, // OAuth users don't have passwords
        emailVerified: new Date(),
      })
      .returning();

    return newUser;
  } catch (error) {
    console.error('OAUTH_UPSERT_ERROR:', error);
    console.error('OAUTH_USER_DATA:', {
      email,
      name,
      imageLength: image?.length,
    });
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to upsert OAuth user',
    );
  }
}

export async function saveChat({
  id,
  userId,
  title,
  visibility,
}: {
  id: string;
  userId: string;
  title: string;
  visibility: VisibilityType;
}) {
  try {
    return await db.insert(chat).values({
      id,
      createdAt: new Date(),
      userId,
      title,
      visibility,
    });
  } catch (error) {
    console.error('[saveChat] DB error:', error);
    throw new ChatSDKError('bad_request:database', 'Failed to save chat');
  }
}

export async function deleteChatById({ id }: { id: string }) {
  try {
    await db.delete(vote).where(eq(vote.chatId, id));
    await db.delete(message).where(eq(message.chatId, id));
    await db.delete(stream).where(eq(stream.chatId, id));

    const [chatsDeleted] = await db
      .delete(chat)
      .where(eq(chat.id, id))
      .returning();
    return chatsDeleted;
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to delete chat by id',
    );
  }
}

export async function getChatsByUserId({
  id,
  limit,
  startingAfter,
  endingBefore,
}: {
  id: string;
  limit: number;
  startingAfter: string | null;
  endingBefore: string | null;
}) {
  try {
    const extendedLimit = limit + 1;

    const query = (whereCondition?: SQL<any>) =>
      db
        .select()
        .from(chat)
        .where(
          whereCondition
            ? and(whereCondition, eq(chat.userId, id))
            : eq(chat.userId, id),
        )
        .orderBy(desc(chat.createdAt))
        .limit(extendedLimit);

    let filteredChats: Array<Chat> = [];

    if (startingAfter) {
      const [selectedChat] = await db
        .select()
        .from(chat)
        .where(eq(chat.id, startingAfter))
        .limit(1);

      if (!selectedChat) {
        throw new ChatSDKError(
          'not_found:database',
          `Chat with id ${startingAfter} not found`,
        );
      }

      filteredChats = await query(gt(chat.createdAt, selectedChat.createdAt));
    } else if (endingBefore) {
      const [selectedChat] = await db
        .select()
        .from(chat)
        .where(eq(chat.id, endingBefore))
        .limit(1);

      if (!selectedChat) {
        throw new ChatSDKError(
          'not_found:database',
          `Chat with id ${endingBefore} not found`,
        );
      }

      filteredChats = await query(lt(chat.createdAt, selectedChat.createdAt));
    } else {
      filteredChats = await query();
    }

    const hasMore = filteredChats.length > limit;

    return {
      chats: hasMore ? filteredChats.slice(0, limit) : filteredChats,
      hasMore,
    };
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get chats by user id',
    );
  }
}

export async function getChatById({ id }: { id: string }) {
  try {
    const [selectedChat] = await db.select().from(chat).where(eq(chat.id, id));
    return selectedChat;
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to get chat by id');
  }
}

export async function saveMessages({
  messages,
}: {
  messages: Array<DBMessage>;
}) {
  try {
    return await db.insert(message).values(messages);
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to save messages');
  }
}

export async function getMessagesByChatId({ id }: { id: string }) {
  try {
    return await db
      .select()
      .from(message)
      .where(eq(message.chatId, id))
      .orderBy(asc(message.createdAt));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get messages by chat id',
    );
  }
}

export async function voteMessage({
  chatId,
  messageId,
  type,
}: {
  chatId: string;
  messageId: string;
  type: 'up' | 'down';
}) {
  try {
    const [existingVote] = await db
      .select()
      .from(vote)
      .where(and(eq(vote.messageId, messageId)));

    if (existingVote) {
      return await db
        .update(vote)
        .set({ isUpvoted: type === 'up' })
        .where(and(eq(vote.messageId, messageId), eq(vote.chatId, chatId)));
    }
    return await db.insert(vote).values({
      chatId,
      messageId,
      isUpvoted: type === 'up',
    });
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to vote message');
  }
}

export async function getVotesByChatId({ id }: { id: string }) {
  try {
    return await db.select().from(vote).where(eq(vote.chatId, id));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get votes by chat id',
    );
  }
}

export async function saveDocument({
  id,
  title,
  kind,
  content,
  userId,
}: {
  id: string;
  title: string;
  kind: ArtifactKind;
  content: string;
  userId: string;
}) {
  try {
    return await db
      .insert(document)
      .values({
        id,
        title,
        kind,
        content,
        userId,
        createdAt: new Date(),
      })
      .returning();
  } catch (error) {
    console.error('Database error saving document:', error);
    console.error('Error details:', JSON.stringify(error, null, 2));
    throw new ChatSDKError(
      'bad_request:database',
      `Failed to save document: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

export async function getDocumentsById({ id }: { id: string }) {
  try {
    const documents = await db
      .select()
      .from(document)
      .where(eq(document.id, id))
      .orderBy(asc(document.createdAt));

    return documents;
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get documents by id',
    );
  }
}

export async function getDocumentById({ id }: { id: string }) {
  try {
    const [selectedDocument] = await db
      .select()
      .from(document)
      .where(eq(document.id, id))
      .orderBy(desc(document.createdAt));

    return selectedDocument;
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get document by id',
    );
  }
}

export async function deleteDocumentsByIdAfterTimestamp({
  id,
  timestamp,
}: {
  id: string;
  timestamp: Date;
}) {
  try {
    await db
      .delete(suggestion)
      .where(
        and(
          eq(suggestion.documentId, id),
          gt(suggestion.documentCreatedAt, timestamp),
        ),
      );

    return await db
      .delete(document)
      .where(and(eq(document.id, id), gt(document.createdAt, timestamp)))
      .returning();
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to delete documents by id after timestamp',
    );
  }
}

export async function saveSuggestions({
  suggestions,
}: {
  suggestions: Array<Suggestion>;
}) {
  try {
    return await db.insert(suggestion).values(suggestions);
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to save suggestions',
    );
  }
}

export async function getSuggestionsByDocumentId({
  documentId,
}: {
  documentId: string;
}) {
  try {
    return await db
      .select()
      .from(suggestion)
      .where(and(eq(suggestion.documentId, documentId)));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get suggestions by document id',
    );
  }
}

export async function getMessageById({ id }: { id: string }) {
  try {
    return await db.select().from(message).where(eq(message.id, id));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get message by id',
    );
  }
}

export async function deleteMessagesByChatIdAfterTimestamp({
  chatId,
  timestamp,
}: {
  chatId: string;
  timestamp: Date;
}) {
  try {
    const messagesToDelete = await db
      .select({ id: message.id })
      .from(message)
      .where(
        and(eq(message.chatId, chatId), gte(message.createdAt, timestamp)),
      );

    const messageIds = messagesToDelete.map((message) => message.id);

    if (messageIds.length > 0) {
      await db
        .delete(vote)
        .where(
          and(eq(vote.chatId, chatId), inArray(vote.messageId, messageIds)),
        );

      return await db
        .delete(message)
        .where(
          and(eq(message.chatId, chatId), inArray(message.id, messageIds)),
        );
    }
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to delete messages by chat id after timestamp',
    );
  }
}

export async function updateChatVisiblityById({
  chatId,
  visibility,
}: {
  chatId: string;
  visibility: 'private' | 'public';
}) {
  try {
    return await db.update(chat).set({ visibility }).where(eq(chat.id, chatId));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to update chat visibility by id',
    );
  }
}

export async function getMessageCountByUserId({
  id,
  differenceInHours,
}: { id: string; differenceInHours: number }) {
  try {
    const twentyFourHoursAgo = new Date(
      Date.now() - differenceInHours * 60 * 60 * 1000,
    );

    const [stats] = await db
      .select({ count: count(message.id) })
      .from(message)
      .innerJoin(chat, eq(message.chatId, chat.id))
      .where(
        and(
          eq(chat.userId, id),
          gte(message.createdAt, twentyFourHoursAgo),
          eq(message.role, 'user'),
        ),
      )
      .execute();

    return stats?.count ?? 0;
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get message count by user id',
    );
  }
}

export async function createStreamId({
  streamId,
  chatId,
}: {
  streamId: string;
  chatId: string;
}) {
  try {
    await db
      .insert(stream)
      .values({ id: streamId, chatId, createdAt: new Date() });
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to create stream id',
    );
  }
}

export async function getStreamIdsByChatId({ chatId }: { chatId: string }) {
  try {
    const streamIds = await db
      .select({ id: stream.id })
      .from(stream)
      .where(eq(stream.chatId, chatId))
      .orderBy(asc(stream.createdAt))
      .execute();

    return streamIds.map(({ id }) => id);
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get stream ids by chat id',
    );
  }
}

/**
 * Upsert the session mapping for a chat. Populated from two directions:
 * the client sends posthogSessionId, the server sends kernelSessionId/
 * kernelReplayId. One row per chat (unique on chatId) — only the provided,
 * non-undefined id fields are written, so neither side clobbers the other's.
 */
export async function upsertSessionMapping({
  chatId,
  userId,
  posthogSessionId,
  posthogReplayUrl,
  kernelSessionId,
  kernelReplayId,
  kernelReplayUrl,
}: {
  chatId: string;
  userId: string;
  posthogSessionId?: string;
  posthogReplayUrl?: string;
  kernelSessionId?: string;
  kernelReplayId?: string;
  kernelReplayUrl?: string;
}) {
  try {
    const now = new Date();

    // Only the fields actually provided are updated on conflict, so a later
    // write from one source never nulls out a value set by the other.
    const setOnConflict: Record<string, unknown> = { updatedAt: now };
    if (posthogSessionId !== undefined)
      setOnConflict.posthogSessionId = posthogSessionId;
    if (posthogReplayUrl !== undefined)
      setOnConflict.posthogReplayUrl = posthogReplayUrl;
    if (kernelSessionId !== undefined)
      setOnConflict.kernelSessionId = kernelSessionId;
    if (kernelReplayId !== undefined)
      setOnConflict.kernelReplayId = kernelReplayId;
    if (kernelReplayUrl !== undefined)
      setOnConflict.kernelReplayUrl = kernelReplayUrl;

    return await db
      .insert(sessionMapping)
      .values({
        chatId,
        userId,
        posthogSessionId,
        posthogReplayUrl,
        kernelSessionId,
        kernelReplayId,
        kernelReplayUrl,
        createdAt: now,
        updatedAt: now,
      })
      .onConflictDoUpdate({
        target: sessionMapping.chatId,
        set: setOnConflict,
      });
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to upsert session mapping',
    );
  }
}

// EveRun: per-(user, chat) Eve run state. Every query that takes a userId is
// scoped by it. swapToken and advance are single conditional UPDATEs so
// concurrent writers cannot interleave a read and a write.

export async function getEveRun({
  userId,
  chatId,
}: { userId: string; chatId: string }): Promise<EveRun | undefined> {
  try {
    const [row] = await db
      .select()
      .from(eveRun)
      .where(and(eq(eveRun.userId, userId), eq(eveRun.chatId, chatId)));
    return row;
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to get eve run');
  }
}

/**
 * Looks a run up by chatId alone, for the internal token route. chatId is
 * only unique per user, so if two users hold the same chatId this fails
 * closed and returns undefined rather than picking one.
 */
export async function getEveRunByChatId({
  chatId,
}: { chatId: string }): Promise<EveRun | undefined> {
  try {
    const rows = await db
      .select()
      .from(eveRun)
      .where(eq(eveRun.chatId, chatId))
      .limit(2);
    return rows.length === 1 ? rows[0] : undefined;
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to get eve run by chat id',
    );
  }
}

/** Inserts a new run; throws if one already exists for (userId, chatId). */
export async function createEveRun({
  userId,
  chatId,
  eveSessionId,
  continuationToken,
  relayOrigin,
}: {
  userId: string;
  chatId: string;
  eveSessionId: string;
  continuationToken: string;
  relayOrigin?: string | null;
}): Promise<EveRun> {
  try {
    const now = new Date();
    const [row] = await db
      .insert(eveRun)
      .values({
        userId,
        chatId,
        eveSessionId,
        continuationToken,
        relayOrigin: relayOrigin ?? null,
        createdAt: now,
        updatedAt: now,
      })
      .returning();
    return row;
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to create eve run');
  }
}

/** Compare-and-set on continuationToken. Returns whether the swap happened. */
export async function swapEveRunToken({
  userId,
  chatId,
  expected,
  next,
}: {
  userId: string;
  chatId: string;
  expected: string;
  next: string;
}): Promise<boolean> {
  try {
    const rows = await db
      .update(eveRun)
      .set({ continuationToken: next, updatedAt: new Date() })
      .where(
        and(
          eq(eveRun.userId, userId),
          eq(eveRun.chatId, chatId),
          eq(eveRun.continuationToken, expected),
        ),
      )
      .returning({ id: eveRun.id });
    return rows.length === 1;
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to swap eve run token',
    );
  }
}

/** Moves the cursor and token forward only when streamIndex is higher. */
export async function advanceEveRun({
  userId,
  chatId,
  continuationToken,
  streamIndex,
}: {
  userId: string;
  chatId: string;
  continuationToken: string;
  streamIndex: number;
}): Promise<void> {
  try {
    await db
      .update(eveRun)
      .set({ continuationToken, streamIndex, updatedAt: new Date() })
      .where(
        and(
          eq(eveRun.userId, userId),
          eq(eveRun.chatId, chatId),
          lt(eveRun.streamIndex, streamIndex),
        ),
      );
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to advance eve run');
  }
}

export async function setEveRunPending({
  userId,
  chatId,
  pendingRequests,
}: {
  userId: string;
  chatId: string;
  pendingRequests: PendingInputRequest[];
}): Promise<void> {
  try {
    await db
      .update(eveRun)
      .set({ pendingRequests, updatedAt: new Date() })
      .where(and(eq(eveRun.userId, userId), eq(eveRun.chatId, chatId)));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to set eve run pending requests',
    );
  }
}

export async function setEveRunProgress({
  userId,
  chatId,
  progress,
}: {
  userId: string;
  chatId: string;
  progress: RunProgress;
}): Promise<void> {
  try {
    await db
      .update(eveRun)
      .set({ progress, updatedAt: new Date() })
      .where(and(eq(eveRun.userId, userId), eq(eveRun.chatId, chatId)));
  } catch (error) {
    throw new ChatSDKError(
      'bad_request:database',
      'Failed to set eve run progress',
    );
  }
}

export async function deleteEveRun({
  userId,
  chatId,
}: { userId: string; chatId: string }): Promise<void> {
  try {
    await db
      .delete(eveRun)
      .where(and(eq(eveRun.userId, userId), eq(eveRun.chatId, chatId)));
  } catch (error) {
    throw new ChatSDKError('bad_request:database', 'Failed to delete eve run');
  }
}
