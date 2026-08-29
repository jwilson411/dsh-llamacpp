import assert from 'node:assert/strict'
import http from 'node:http'
import test from 'node:test'

import { LlmError, userAgent } from '@deepseek-ai/dsh-llm'

import { LlamaCppAdapter } from '../src/adapter.js'

/**
 * Start a localhost-only mock of the llama.cpp server. No real llama-server,
 * no weights, no network beyond the loopback interface.
 * @param handler - request handler, given the recorded request and the response.
 * @returns the base URL, the recorded requests, and a close function.
 */
async function startMockServer(handler) {
  const requests = []
  const server = http.createServer((req, res) => {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
    })
    req.on('end', () => {
      const record = { method: req.method, url: req.url, headers: req.headers, body }
      requests.push(record)
      handler(record, res)
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const { port } = server.address()
  return {
    baseURL: `http://127.0.0.1:${port}/v1`,
    requests,
    close: () => new Promise((resolve) => server.close(resolve)),
  }
}

/**
 * Reserve a port and then release it, so connecting to it is refused.
 * @returns a base URL nothing is listening on.
 */
async function closedBaseURL() {
  const server = http.createServer()
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const { port } = server.address()
  await new Promise((resolve) => server.close(resolve))
  return `http://127.0.0.1:${port}/v1`
}

/**
 * Write an SSE response body.
 * @param res - the mock server response.
 * @param frames - `data:` payloads to send, in order.
 */
function sendSse(res, frames) {
  res.writeHead(200, { 'content-type': 'text/event-stream' })
  for (const frame of frames) res.write(`data: ${frame}\n\n`)
  res.end()
}

/**
 * Drain an async iterable into an array.
 * @param stream - the chunk stream.
 * @returns every chunk, in order.
 */
async function collect(stream) {
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

/**
 * A minimal request in the harness shape.
 * @param overrides - fields to merge into the request.
 * @returns the assembled options.
 */
const request = (overrides = {}) => ({
  provider: 'llamacpp',
  model: '',
  messages: [{ id: 'm1', role: 'user', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }],
  ...overrides,
})

test('stream posts to the configured endpoint and yields the harness chunk sequence', async (t) => {
  const mock = await startMockServer((_req, res) =>
    sendSse(res, [
      '{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
      '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
      '{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
      '{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2}}',
      '[DONE]',
    ]),
  )
  t.after(() => mock.close())

  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen', apiKey: 'k' })
  const chunks = await collect(adapter.stream(request({ system: 'be terse' })))

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hello' },
    { type: 'text-delta', index: 0, text: ' world' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello world' } },
    { type: 'usage', usage: { inputTokens: 9, outputTokens: 2 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])

  const [recorded] = mock.requests
  assert.equal(recorded.method, 'POST')
  assert.equal(recorded.url, '/v1/chat/completions')
  assert.equal(recorded.headers['user-agent'], userAgent())
  assert.equal(recorded.headers.authorization, 'Bearer k')
  const body = JSON.parse(recorded.body)
  assert.equal(body.stream, true)
  assert.equal(body.model, 'qwen')
  assert.deepEqual(body.messages, [
    { role: 'system', content: 'be terse' },
    { role: 'user', content: 'hi' },
  ])
})

test('stream sends no authorization header when no api key is configured', async (t) => {
  const mock = await startMockServer((_req, res) => sendSse(res, ['[DONE]']))
  t.after(() => mock.close())

  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen' })
  await collect(adapter.stream(request()))

  assert.equal('authorization' in mock.requests[0].headers, false)
})

test('stream prefers the request model over the configured one', async (t) => {
  const mock = await startMockServer((_req, res) => sendSse(res, ['[DONE]']))
  t.after(() => mock.close())

  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen' })
  await collect(adapter.stream(request({ model: 'qwen3-coder' })))

  assert.equal(JSON.parse(mock.requests[0].body).model, 'qwen3-coder')
})

test('stream translates a non-streaming JSON answer into the same sequence', async (t) => {
  const mock = await startMockServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ choices: [{ message: { content: 'Hi' }, finish_reason: 'stop' }] }))
  })
  t.after(() => mock.close())

  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen' })
  assert.deepEqual(await collect(adapter.stream(request())), [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hi' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hi' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('stream forwards options.signal so an abort reaches the server request', async (t) => {
  const mock = await startMockServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    res.write('data: {"choices":[{"delta":{"content":"one"},"finish_reason":null}]}\n\n')
    // Deliberately left open: only the abort ends this request.
  })
  t.after(() => mock.close())

  const controller = new AbortController()
  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen' })
  const iterator = adapter.stream(request({ signal: controller.signal }))[Symbol.asyncIterator]()

  assert.deepEqual((await iterator.next()).value, { type: 'block-start', index: 0, blockType: 'text' })
  assert.deepEqual((await iterator.next()).value, { type: 'text-delta', index: 0, text: 'one' })
  controller.abort()
  await assert.rejects(iterator.next(), (error) => {
    assert.equal(controller.signal.aborted, true)
    assert.equal(error instanceof LlmError, false)
    return true
  })
})

test('stream throws PROVIDER_UNREACHABLE when nothing is listening', async () => {
  const adapter = new LlamaCppAdapter({ baseURL: await closedBaseURL(), model: 'qwen' })
  await assert.rejects(collect(adapter.stream(request())), (error) => {
    assert.ok(error instanceof LlmError)
    assert.equal(error.code, 'PROVIDER_UNREACHABLE')
    assert.match(error.message, /llama-server/)
    return true
  })
})

test('stream throws PROVIDER_UNREACHABLE when the host does not resolve', async () => {
  const adapter = new LlamaCppAdapter({ baseURL: 'http://llamacpp.invalid:8080/v1', model: 'qwen' })
  await assert.rejects(collect(adapter.stream(request())), (error) => {
    assert.equal(error.code, 'PROVIDER_UNREACHABLE')
    return true
  })
})

test('stream throws PROVIDER_HTTP_ERROR carrying the status on a non-2xx answer', async (t) => {
  const mock = await startMockServer((_req, res) => {
    res.writeHead(503, { 'content-type': 'application/json' })
    res.end('{"error":{"message":"model is loading"}}')
  })
  t.after(() => mock.close())

  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen' })
  await assert.rejects(collect(adapter.stream(request())), (error) => {
    assert.ok(error instanceof LlmError)
    assert.equal(error.code, 'PROVIDER_HTTP_ERROR')
    assert.equal(error.failure.status, 503)
    assert.match(error.message, /model is loading/)
    return true
  })
})

test('stream surfaces an in-band error frame instead of finishing normally', async (t) => {
  const mock = await startMockServer((_req, res) =>
    sendSse(res, ['{"error":{"message":"context window exceeded"}}']),
  )
  t.after(() => mock.close())

  const adapter = new LlamaCppAdapter({ baseURL: mock.baseURL, model: 'qwen' })
  await assert.rejects(collect(adapter.stream(request())), (error) => {
    assert.equal(error.code, 'PROVIDER_ERROR')
    return true
  })
})

test('providerInfo and listModels describe the configured route and model', async () => {
  const adapter = new LlamaCppAdapter({ baseURL: 'http://127.0.0.1:8080/v1', model: 'qwen' })
  assert.deepEqual(adapter.providerInfo('llamacpp'), { id: 'llamacpp', name: 'llama.cpp' })
  assert.deepEqual(await adapter.listModels('llamacpp'), [
    { provider: 'llamacpp', id: 'qwen', name: 'qwen', inputModalities: ['text'] },
  ])
})
