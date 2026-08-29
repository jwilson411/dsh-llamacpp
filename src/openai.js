/**
 * Pure translation between the harness request/stream vocabulary and the
 * OpenAI-compatible wire format llama.cpp's server speaks. Nothing here does
 * I/O, so every mapping decision is directly testable.
 *
 * @module dsh-llamacpp/openai
 */
import { LlmError, attributionHeaders } from '@deepseek-ai/dsh-llm'

/** A message content block this thin adapter cannot represent on the wire. */
export const UNSUPPORTED_CONTENT_CODE = 'UNSUPPORTED_CONTENT'

/** The server answered, but its stream framing or payload was not parseable. */
export const PROVIDER_PROTOCOL_ERROR_CODE = 'PROVIDER_PROTOCOL_ERROR'

/** The server reported a failure in-band (an SSE `error` payload). */
export const PROVIDER_ERROR_CODE = 'PROVIDER_ERROR'

/**
 * Build the chat-completions endpoint for a configured base URL. The default
 * base URL already ends in `/v1`, so the `/v1` is never added a second time.
 * @param baseURL - configured server base, with or without a trailing slash.
 * @returns the absolute URL to POST a chat completion to.
 */
export function chatCompletionsUrl(baseURL) {
  return `${String(baseURL).replace(/\/+$/, '')}/chat/completions`
}

/**
 * Headers for one provider request. Attribution is merged first so a caller
 * cannot accidentally drop it, and the credential is sent only when set.
 * @param apiKey - the llama.cpp `--api-key` value, when one is configured.
 * @returns lowercase header names ready for `fetch`.
 */
export function requestHeaders(apiKey) {
  const headers = {
    ...attributionHeaders(),
    'content-type': 'application/json',
    accept: 'text/event-stream',
  }
  if (apiKey) headers.authorization = `Bearer ${apiKey}`
  return headers
}

/**
 * Flatten one harness message's content blocks into the single string
 * OpenAI-compatible endpoints accept. Image and tool blocks fail loud: this
 * adapter sends text, and silently dropping a block would hand the model a
 * conversation the caller never wrote.
 * @param message - one harness message.
 * @returns the concatenated text of its text blocks.
 */
function messageText(message) {
  let text = ''
  for (const block of message.content ?? []) {
    if (block.type === 'text') {
      text += block.text
      continue
    }
    throw new LlmError(
      `dsh-llamacpp: cannot send a "${block.type}" content block to llama.cpp; this adapter is text-only`,
      UNSUPPORTED_CONTENT_CODE,
    )
  }
  return text
}

/**
 * Map the harness conversation to OpenAI `{ role, content }` messages, with
 * `options.system` prepended as the system slot.
 * @param options - the assembled {@link GenerateOptions}.
 * @returns wire messages in conversation order.
 */
export function toOpenAiMessages(options) {
  const messages = []
  if (options.system) messages.push({ role: 'system', content: options.system })
  for (const message of options.messages ?? []) {
    messages.push({ role: message.role, content: messageText(message) })
  }
  return messages
}

/**
 * Map harness tool schemas to the OpenAI `tools` array.
 * @param tools - schemas from {@link GenerateOptions.tools}.
 * @returns the wire tools array, or undefined when none were requested.
 */
export function toOpenAiTools(tools) {
  if (!tools?.length) return undefined
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))
}

/**
 * Build the streaming chat-completion request body.
 *
 * `maxTokens` maps to `max_tokens`: llama.cpp's server does not read
 * `max_completion_tokens`.
 * @param options - the assembled {@link GenerateOptions}.
 * @param fallbackModel - configured model id, used when the request omits one.
 * @returns the JSON body to POST.
 */
export function buildRequestBody(options, fallbackModel) {
  const body = {
    model: options.model || fallbackModel,
    messages: toOpenAiMessages(options),
    stream: true,
  }
  const tools = toOpenAiTools(options.tools)
  if (tools) body.tools = tools
  if (options.temperature !== undefined) body.temperature = options.temperature
  if (options.maxTokens !== undefined) body.max_tokens = options.maxTokens
  if (options.stop?.length) body.stop = [...options.stop]
  return body
}

/**
 * Map an OpenAI `finish_reason` to the harness finish vocabulary. `length` is
 * reported as `max-tokens` rather than `stop`, so a truncated answer is not
 * presented as a complete one.
 * @param reason - the provider's `finish_reason`, if it sent one.
 * @returns the harness {@link FinishReason}.
 */
export function toFinishReason(reason) {
  if (reason === 'tool_calls') return { kind: 'tool-calls' }
  if (reason === 'length') return { kind: 'max-tokens' }
  return { kind: 'stop' }
}

/**
 * Map an OpenAI usage object to harness {@link TokenUsage}. llama.cpp reports
 * no cache split, so no cache fields are invented.
 * @param usage - the provider's `usage` object, if it sent one.
 * @returns harness usage, or undefined when the counts are absent.
 */
export function toTokenUsage(usage) {
  if (!usage) return undefined
  const inputTokens = usage.prompt_tokens
  const outputTokens = usage.completion_tokens
  if (typeof inputTokens !== 'number' && typeof outputTokens !== 'number') return undefined
  return { inputTokens: inputTokens ?? 0, outputTokens: outputTokens ?? 0 }
}

/**
 * Incremental SSE framing. Feed it decoded response text; it returns the
 * `data:` payloads of every event completed so far, in order.
 * @returns a parser with `push(text)` and `flush()`.
 */
export function createSseParser() {
  let buffer = ''
  let data = []
  const payloads = []

  const dispatch = () => {
    if (data.length === 0) return
    payloads.push(data.join('\n'))
    data = []
  }

  const consume = (line) => {
    if (line === '') {
      dispatch()
      return
    }
    if (line.startsWith(':')) return
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    if (field !== 'data') return
    const value = colon === -1 ? '' : line.slice(colon + 1)
    data.push(value.startsWith(' ') ? value.slice(1) : value)
  }

  return {
    /**
     * Accept the next decoded chunk of the response body.
     * @param text - decoded bytes, split anywhere.
     * @returns complete event payloads, in arrival order.
     */
    push(text) {
      buffer += text
      let index
      while ((index = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, index).replace(/\r$/, '')
        buffer = buffer.slice(index + 1)
        consume(line)
      }
      return payloads.splice(0, payloads.length)
    },
    /**
     * Close the stream, dispatching any event the server left unterminated.
     * @returns the remaining event payloads.
     */
    flush() {
      if (buffer.length > 0) {
        consume(buffer.replace(/\r$/, ''))
        buffer = ''
      }
      dispatch()
      return payloads.splice(0, payloads.length)
    },
  }
}

/**
 * Assemble harness {@link StreamChunk}s from OpenAI chat-completion frames.
 *
 * The emitted sequence is: `block-start` (index 0, `text`), one `text-delta`
 * per non-empty content delta, `block-end` carrying the full assembled text,
 * `usage` when the server reported any, then a terminal `finish`. A response
 * that produced no text at all starts no block, so every `block-start` still
 * has its `block-end`.
 * @returns a translator with `accept(payload)` and `end()`.
 */
export function createChunkTranslator() {
  let started = false
  let text = ''
  let usage
  let finishReason

  return {
    /**
     * Translate one decoded streaming frame.
     * @param frame - a parsed `chat.completion.chunk` object.
     * @returns the chunks this frame produced.
     */
    accept(frame) {
      const chunks = []
      const usageUpdate = toTokenUsage(frame?.usage)
      if (usageUpdate) usage = usageUpdate
      const choice = frame?.choices?.[0]
      if (!choice) return chunks
      if (choice.finish_reason) finishReason = choice.finish_reason
      const content = choice.delta?.content
      if (typeof content !== 'string' || content.length === 0) return chunks
      if (!started) {
        started = true
        chunks.push({ type: 'block-start', index: 0, blockType: 'text' })
      }
      text += content
      chunks.push({ type: 'text-delta', index: 0, text: content })
      return chunks
    },

    /**
     * Close the response.
     * @returns the trailing `block-end`, `usage`, and `finish` chunks.
     */
    end() {
      const chunks = []
      if (started) chunks.push({ type: 'block-end', index: 0, block: { type: 'text', text } })
      if (usage) chunks.push({ type: 'usage', usage })
      chunks.push({ type: 'finish', reason: toFinishReason(finishReason) })
      return chunks
    },
  }
}

/**
 * Decode one SSE payload, failing loud on anything that is not a frame.
 * @param payload - the raw `data:` value, excluding the `[DONE]` sentinel.
 * @returns the parsed frame.
 */
export function parseFrame(payload) {
  let frame
  try {
    frame = JSON.parse(payload)
  } catch (cause) {
    throw new LlmError(
      'dsh-llamacpp: llama.cpp sent a stream frame that is not JSON',
      PROVIDER_PROTOCOL_ERROR_CODE,
      { cause },
    )
  }
  assertNoProviderError(frame)
  return frame
}

/**
 * Reject a payload in which the server reported its own failure in-band.
 * @param frame - a decoded response or stream frame.
 */
export function assertNoProviderError(frame) {
  const error = frame?.error
  if (!error) return
  const message = typeof error === 'string' ? error : (error.message ?? JSON.stringify(error))
  throw new LlmError(`dsh-llamacpp: llama.cpp reported an error: ${message}`, PROVIDER_ERROR_CODE)
}

/**
 * Translate a non-streaming `chat.completion` response into the same chunk
 * sequence a stream would have produced. Some llama.cpp builds and proxies
 * ignore `stream: true` and answer with one JSON object.
 * @param response - the decoded `chat.completion` body.
 * @returns the full chunk sequence, terminal `finish` last.
 */
export function chunksFromCompletion(response) {
  assertNoProviderError(response)
  const translator = createChunkTranslator()
  const choice = response?.choices?.[0]
  const chunks = translator.accept({
    choices: [{ delta: { content: choice?.message?.content ?? '' }, finish_reason: choice?.finish_reason }],
    usage: response?.usage,
  })
  return [...chunks, ...translator.end()]
}
