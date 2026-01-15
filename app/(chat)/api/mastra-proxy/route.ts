// Web automation proxy - routes to the new AI SDK based web-automation endpoint
// This keeps backward compatibility with the existing client code

export const maxDuration = 300; // 5 minutes for web automation tasks

// GET /api/mastra-proxy?action=browser-session&sessionId=xxx
// Browser session info is no longer needed as we send liveViewUrl via stream
export async function GET(request: Request) {
  const url = new URL(request.url);
  const action = url.searchParams.get('action');
  const sessionId = url.searchParams.get('sessionId');

  if (action === 'browser-session' && sessionId) {
    // Return a placeholder - the actual liveViewUrl comes via the stream
    return new Response(
      JSON.stringify({
        sessionId,
        message: 'Live view URL is sent via the chat stream',
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  }

  return new Response(
    JSON.stringify({ error: 'Invalid action or missing parameters' }),
    { status: 400, headers: { 'Content-Type': 'application/json' } }
  );
}

export async function POST(request: Request) {
  try {
    const body = await request.json();

    // Check if the request is to stop the chat
    if (body.action === 'stopChat') {
      console.log('[mastra-proxy] Stopping chat for thread:', body.threadId, 'and resource:', body.resourceId);

      // Call the web-automation DELETE endpoint to stop the session
      const baseUrl = new URL(request.url).origin;
      const stopResponse = await fetch(
        `${baseUrl}/api/web-automation?threadId=${encodeURIComponent(body.threadId)}&resourceId=${encodeURIComponent(body.resourceId)}`,
        { method: 'DELETE' }
      );

      const data = await stopResponse.json();
      return new Response(JSON.stringify(data), {
        status: stopResponse.status,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Forward chat requests to the new web-automation endpoint
    console.log('[mastra-proxy] Forwarding to web-automation endpoint');

    const baseUrl = new URL(request.url).origin;
    const response = await fetch(`${baseUrl}/api/web-automation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Forward auth headers
        cookie: request.headers.get('cookie') || '',
      },
      body: JSON.stringify(body),
    });

    // Return the response as-is (preserves streaming)
    return new Response(response.body, {
      status: response.status,
      headers: response.headers,
    });
  } catch (error) {
    console.error('[mastra-proxy] Error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process web automation request' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
