export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

export async function classifyIntent(question) {
  const response = await fetch(`${API_URL}/classify-intent`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error')
    throw new Error(`Failed to classify intent: ${errorText}`)
  }

  const data = await response.json()
  if (!data || typeof data !== 'object' || !data.intent) {
    throw new Error('Invalid intent response format')
  }
  return data
}

export async function askQuestion(question) {
  // PDF Q/A ONLY: Classify intent first (UI rendering + backend routing)
  const intentInfo = await classifyIntent(question)

  // PDF Q/A ONLY: Call /ask endpoint
  const response = await fetch(`${API_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  });
  
  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error')
    throw new Error(`Failed to get answer: ${errorText}`);
  }
  
  const data = await response.json();

  // Allow backend error payloads (e.g., exact reference not found)
  if (data && typeof data === 'object' && data.error) {
    return {
      error: data.error,
      intent: intentInfo.intent,
      target_id: intentInfo.target_id ?? null,
      blocks: [],
      sources: []
    }
  }
  
  // STRICT VALIDATION: Ensure response has PDF Q/A structure (blocks, sources)
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid response format: expected object with blocks and sources');
  }
  
  if (!Array.isArray(data.blocks)) {
    throw new Error('Invalid response format: blocks must be an array');
  }
  
  if (!Array.isArray(data.sources)) {
    throw new Error('Invalid response format: sources must be an array');
  }
  
  return {
    ...data,
    intent: intentInfo.intent,
    target_id: intentInfo.target_id ?? null
  };
}

export async function expandContext(chunkIds, intent, targetId) {
  const response = await fetch(`${API_URL}/expand-context`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      chunk_ids: chunkIds,
      intent: intent || null,
      target_id: targetId || null,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error')
    throw new Error(`Failed to expand context: ${errorText}`)
  }

  const data = await response.json()
  if (!data || typeof data !== 'object' || !Array.isArray(data.contextual_sources)) {
    throw new Error('Invalid context expansion response format')
  }
  return data.contextual_sources
}

/** Generate a short chat title from the first question. Uses backend only; no API key in frontend. */
export async function generateChatTitle(question) {
  const response = await fetch(`${API_URL}/generate-chat-title`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error')
    throw new Error(`Failed to generate title: ${errorText}`)
  }

  const data = await response.json()
  if (data && typeof data.error === 'string') {
    throw new Error(data.error)
  }
  if (!data || typeof data.title !== 'string') {
    throw new Error('Invalid generate-chat-title response')
  }
  return data.title
}

