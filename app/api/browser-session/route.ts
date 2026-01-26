import { auth } from '@/app/(auth)/auth';
import {
  deleteBrowserSession,
  getBrowserSession,
} from '@/lib/browser/kernel-service';

export async function DELETE(request: Request) {
  const session = await auth();
  if (!session?.user) {
    return new Response('Unauthorized', { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const chatId = searchParams.get('chatId');

  if (!chatId) {
    return new Response('Missing chatId parameter', { status: 400 });
  }

  try {
    await deleteBrowserSession(chatId);
    return new Response('OK', { status: 200 });
  } catch (error) {
    console.error('Failed to delete browser session:', error);
    return new Response('Failed to delete browser session', { status: 500 });
  }
}

export async function GET(request: Request) {
  const session = await auth();
  if (!session?.user) {
    return new Response('Unauthorized', { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const chatId = searchParams.get('chatId');

  if (!chatId) {
    return new Response('Missing chatId parameter', { status: 400 });
  }

  try {
    const browserSession = await getBrowserSession(chatId);
    return Response.json({ session: browserSession });
  } catch (error) {
    console.error('Failed to get browser session:', error);
    return new Response('Failed to get browser session', { status: 500 });
  }
}
