/**
 * The llama.cpp adapter: one HTTP request per model call against the server's
 * OpenAI-compatible `/chat/completions` endpoint, translated into the harness
 * stream vocabulary.
 *
 * @module dsh-llamacpp/adapter
 */
import { LlmAdapter, LlmError, errorChain } from '@deepseek-ai/dsh-llm'
import {
  chatCompletionsUrl,
  chunksFromCompletion,
  createChunkTranslator,
  createSseParser,
  buildRequestBody,
  parseFrame,
  requestHeaders,
} from './openai.js'

/** The server could not be reached at all (refused, DNS failure, reset). */
export const PROVIDER_UNREACHABLE_CODE = 'PROVIDER_UNREACHABLE'

/** The server answered with a non-2xx status. */
export const PROVIDER_HTTP_ERROR_CODE = 'PROVIDER_HTTP_ERROR'

/** Human-readable provider name reported for every route this adapter owns. */
export const PROVIDER_NAME = 'llama.cpp'

/** How much of an error body is quoted back in the failure message. */
const MAX_ERROR_BODY_CHARS = 500

/**
 * Adapter for a locally running `llama-server`. It owns no credentials store,
 * no model catalog beyond the one configured model, and no retry policy: the
 * harness supplies those. Every failure surfaces as an {@link LlmError} with a
 * stable code rather than an empty stream.
 */
export class LlamaCppAdapter extends LlmAdapter {
  #baseURL
  #model
  #apiKey
  #fetch

  /**
   * @param config - resolved configuration from `resolveConfig()`.
   * @param config.baseURL - server base URL, including `/v1`.
   * @param config.model - model id to send when a request omits one.
   * @param config.apiKey - optional `--api-key` credential.
   * @param config.fetch - injectable fetch, for tests.
   */
  constructor({ baseURL, model, apiKey, fetch: fetchImpl } = {}) {
    super()
    this.#baseURL = baseURL
    this.#model = model
    this.#apiKey = apiKey
    this.#fetch = fetchImpl ?? globalThis.fetch
  }

  /** The chat-completions endpoint this adapter posts to. */
  get url() {
    return chatCompletionsUrl(this.#baseURL)
  }

  /**
   * @param provider - a route registered for this adapter.
   * @returns display metadata for that route.
   */
  providerInfo(provider) {
    return { id: provider, name: PROVIDER_NAME }
  }

  /**
   * Advertise the configured model. llama.cpp serves whatever weights the
   * server was started with, named by `--alias`; the catalog is advisory, so
   * an unlisted id passed by the caller is still forwarded.
   * @param provider - a route registered for this adapter.
   * @returns the single configured model entry.
   */
  async listModels(provider) {
    return [{ provider, id: this.#model, name: this.#model, inputModalities: ['text'] }]
  }

  /**
   * Stream one model call.
   * @param options - the assembled request; `options.signal` is forwarded to `fetch`.
   * @yields harness stream chunks, terminal `finish` last.
   */
  async *stream(options) {
    const url = this.url
    const response = await this.#post(url, options)

    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      throw new LlmError(
        `dsh-llamacpp: llama.cpp at ${url} returned HTTP ${response.status}${detail ? `: ${detail.slice(0, MAX_ERROR_BODY_CHARS)}` : ''}`,
        PROVIDER_HTTP_ERROR_CODE,
        { status: response.status },
      )
    }

    // Some builds and proxies ignore `stream: true` and answer with one JSON
    // object; translate it into the same chunk sequence rather than failing.
    if ((response.headers.get('content-type') ?? '').includes('json')) {
      yield* chunksFromCompletion(await response.json())
      return
    }

    yield* this.#streamSse(response, options)
  }

  /**
   * Perform the request, turning transport failures into a stable code.
   * @param url - the chat-completions endpoint.
   * @param options - the assembled request.
   * @returns the raw response.
   */
  async #post(url, options) {
    try {
      return await this.#fetch(url, {
        method: 'POST',
        headers: requestHeaders(this.#apiKey),
        body: JSON.stringify(buildRequestBody(options, this.#model)),
        signal: options.signal,
      })
    } catch (cause) {
      // A caller-driven abort is the caller's own outcome, not a dead server:
      // the runtime turns it into an `aborted` finish.
      if (options.signal?.aborted) throw cause
      throw new LlmError(
        `dsh-llamacpp: cannot reach llama.cpp at ${url} (is llama-server running?): ${errorChain(cause)}`,
        PROVIDER_UNREACHABLE_CODE,
        { cause },
      )
    }
  }

  /**
   * Decode an SSE body into harness chunks.
   * @param response - the streaming response.
   * @param options - the assembled request, for its signal.
   * @yields harness stream chunks.
   */
  async *#streamSse(response, options) {
    const parser = createSseParser()
    const translator = createChunkTranslator()
    const decoder = new TextDecoder()
    let done = false

    const handle = function* (payloads) {
      for (const payload of payloads) {
        if (done) return
        if (payload === '[DONE]') {
          done = true
          return
        }
        yield* translator.accept(parseFrame(payload))
      }
    }

    try {
      for await (const bytes of iterateBody(response, this.url)) {
        yield* handle(parser.push(decoder.decode(bytes, { stream: true })))
        if (done) break
      }
      if (!done) yield* handle(parser.flush())
    } catch (cause) {
      if (options.signal?.aborted || cause instanceof LlmError) throw cause
      throw new LlmError(
        `dsh-llamacpp: llama.cpp stream from ${this.url} ended badly: ${errorChain(cause)}`,
        PROVIDER_UNREACHABLE_CODE,
        { cause },
      )
    }

    yield* translator.end()
  }
}

/**
 * Iterate a response body as byte chunks, refusing a body-less response
 * instead of silently yielding an empty stream.
 * @param response - the streaming response.
 * @param endpoint - endpoint for the diagnostic.
 * @yields raw byte chunks.
 */
async function* iterateBody(response, endpoint) {
  if (!response.body) {
    throw new LlmError(
      `dsh-llamacpp: llama.cpp at ${endpoint} returned no response body`,
      PROVIDER_UNREACHABLE_CODE,
    )
  }
  yield* response.body
}
