import assert from 'node:assert/strict'
import test from 'node:test'

import { LlmError, userAgent } from '@deepseek-ai/dsh-llm'

import {
  buildRequestBody,
  chatCompletionsUrl,
  chunksFromCompletion,
  createChunkTranslator,
  createSseParser,
  parseFrame,
  requestHeaders,
  toFinishReason,
  toOpenAiMessages,
  toOpenAiTools,
  toTokenUsage,
} from '../src/openai.js'

/**
 * A message in the harness shape, without pulling in the id/source machinery
 * the adapter never reads.
 * @param role - conversation role.
 * @param content - content blocks.
 * @returns a message-shaped object.
 */
const message = (role, content) => ({ id: `m-${role}`, role, content, source: { kind: 'user' } })

/**
 * A user message carrying one text block.
 * @param text - the block text.
 * @returns a message-shaped object.
 */
const userText = (text) => message('user', [{ type: 'text', text }])

test('chatCompletionsUrl does not double the /v1 already in the base URL', () => {
  assert.equal(chatCompletionsUrl('http://127.0.0.1:8080/v1'), 'http://127.0.0.1:8080/v1/chat/completions')
  assert.equal(chatCompletionsUrl('http://127.0.0.1:8080/v1/'), 'http://127.0.0.1:8080/v1/chat/completions')
  assert.equal(chatCompletionsUrl('http://localhost:9000'), 'http://localhost:9000/chat/completions')
})

test('requestHeaders always carries attribution and omits an unset credential', () => {
  const headers = requestHeaders(undefined)
  assert.equal(headers['user-agent'], userAgent())
  assert.equal(headers['content-type'], 'application/json')
  assert.equal('authorization' in headers, false)
})

test('requestHeaders sends a bearer credential when one is configured', () => {
  assert.equal(requestHeaders('sekret').authorization, 'Bearer sekret')
})

test('toOpenAiMessages prepends the system slot and flattens text blocks', () => {
  const messages = toOpenAiMessages({
    system: 'be terse',
    messages: [userText('hi '), message('assistant', [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }])],
  })
  assert.deepEqual(messages, [
    { role: 'system', content: 'be terse' },
    { role: 'user', content: 'hi ' },
    { role: 'assistant', content: 'ab' },
  ])
})

test('toOpenAiMessages omits the system slot when there is none', () => {
  assert.deepEqual(toOpenAiMessages({ messages: [userText('hi')] }), [{ role: 'user', content: 'hi' }])
})

test('toOpenAiMessages refuses image content instead of dropping it', () => {
  assert.throws(
    () => toOpenAiMessages({ messages: [message('user', [{ type: 'image', attachment: {} }])] }),
    (error) => {
      assert.ok(error instanceof LlmError)
      assert.equal(error.code, 'UNSUPPORTED_CONTENT')
      return true
    },
  )
})

test('toOpenAiMessages refuses tool blocks this text-only adapter cannot send', () => {
  for (const block of [
    { type: 'tool-call', id: 'c1', name: 't', arguments: '{}' },
    { type: 'tool-result', toolCallId: 'c1', content: [] },
  ]) {
    assert.throws(
      () => toOpenAiMessages({ messages: [message('user', [block])] }),
      (error) => error.code === 'UNSUPPORTED_CONTENT',
    )
  }
})

test('toOpenAiTools maps harness schemas and stays absent when none are sent', () => {
  assert.equal(toOpenAiTools(undefined), undefined)
  assert.equal(toOpenAiTools([]), undefined)
  assert.deepEqual(toOpenAiTools([{ name: 'read', description: 'read a file', parameters: { type: 'object' } }]), [
    { type: 'function', function: { name: 'read', description: 'read a file', parameters: { type: 'object' } } },
  ])
})

test('buildRequestBody streams, uses max_tokens, and forwards the call controls', () => {
  const body = buildRequestBody(
    {
      model: 'qwen3',
      messages: [userText('hi')],
      system: 'sys',
      temperature: 0.2,
      maxTokens: 128,
      stop: ['\n\n'],
      tools: [{ name: 't', description: 'd', parameters: {} }],
    },
    'configured',
  )
  assert.equal(body.model, 'qwen3')
  assert.equal(body.stream, true)
  assert.equal(body.temperature, 0.2)
  assert.equal(body.max_tokens, 128)
  assert.equal(body.max_completion_tokens, undefined)
  assert.deepEqual(body.stop, ['\n\n'])
  assert.equal(body.tools.length, 1)
  assert.deepEqual(body.messages[0], { role: 'system', content: 'sys' })
})

test('buildRequestBody falls back to the configured model and omits absent controls', () => {
  const body = buildRequestBody({ messages: [userText('hi')] }, 'configured')
  assert.equal(body.model, 'configured')
  assert.deepEqual(Object.keys(body).sort(), ['messages', 'model', 'stream'])
})

test('toFinishReason distinguishes stop, tool calls, and truncation', () => {
  assert.deepEqual(toFinishReason('stop'), { kind: 'stop' })
  assert.deepEqual(toFinishReason(undefined), { kind: 'stop' })
  assert.deepEqual(toFinishReason('tool_calls'), { kind: 'tool-calls' })
  assert.deepEqual(toFinishReason('length'), { kind: 'max-tokens' })
})

test('toTokenUsage maps prompt/completion counts and invents no cache split', () => {
  assert.equal(toTokenUsage(undefined), undefined)
  assert.equal(toTokenUsage({}), undefined)
  assert.deepEqual(toTokenUsage({ prompt_tokens: 7, completion_tokens: 3 }), { inputTokens: 7, outputTokens: 3 })
})

test('createSseParser reassembles events split across arbitrary byte chunks', () => {
  const parser = createSseParser()
  assert.deepEqual(parser.push('data: {"a":'), [])
  assert.deepEqual(parser.push('1}\n\ndata: [DO'), ['{"a":1}'])
  assert.deepEqual(parser.push('NE]\n\n'), ['[DONE]'])
  assert.deepEqual(parser.flush(), [])
})

test('createSseParser ignores comments and other SSE fields', () => {
  const parser = createSseParser()
  assert.deepEqual(parser.push(': keep-alive\nevent: message\ndata: {"a":1}\n\n'), ['{"a":1}'])
})

test('createSseParser dispatches an event the server left unterminated', () => {
  const parser = createSseParser()
  assert.deepEqual(parser.push('data: {"a":1}'), [])
  assert.deepEqual(parser.flush(), ['{"a":1}'])
})

test('createChunkTranslator emits the full block-start/deltas/block-end/usage/finish sequence', () => {
  const translator = createChunkTranslator()
  const chunks = [
    ...translator.accept({ choices: [{ delta: { role: 'assistant' }, finish_reason: null }] }),
    ...translator.accept({ choices: [{ delta: { content: 'Hel' }, finish_reason: null }] }),
    ...translator.accept({ choices: [{ delta: { content: 'lo' }, finish_reason: null }] }),
    ...translator.accept({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 2 } }),
    ...translator.end(),
  ]
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hel' },
    { type: 'text-delta', index: 0, text: 'lo' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
    { type: 'usage', usage: { inputTokens: 5, outputTokens: 2 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('createChunkTranslator opens no block when the response carried no text', () => {
  const translator = createChunkTranslator()
  const chunks = [
    ...translator.accept({ choices: [{ delta: { content: '' }, finish_reason: 'stop' }] }),
    ...translator.end(),
  ]
  assert.deepEqual(chunks, [{ type: 'finish', reason: { kind: 'stop' } }])
})

test('createChunkTranslator reports a tool-call finish', () => {
  const translator = createChunkTranslator()
  translator.accept({ choices: [{ delta: { content: 'x' }, finish_reason: 'tool_calls' }] })
  assert.deepEqual(translator.end().at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
})

test('parseFrame fails loud on a non-JSON frame', () => {
  assert.throws(
    () => parseFrame('not json'),
    (error) => error instanceof LlmError && error.code === 'PROVIDER_PROTOCOL_ERROR',
  )
})

test('parseFrame surfaces an in-band provider error', () => {
  assert.throws(
    () => parseFrame('{"error":{"message":"context shift disabled"}}'),
    (error) => {
      assert.equal(error.code, 'PROVIDER_ERROR')
      assert.match(error.message, /context shift disabled/)
      return true
    },
  )
})

test('chunksFromCompletion translates a non-streaming answer into the same sequence', () => {
  assert.deepEqual(
    chunksFromCompletion({
      choices: [{ message: { role: 'assistant', content: 'Hello' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 5, completion_tokens: 2 },
    }),
    [
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'Hello' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
      { type: 'usage', usage: { inputTokens: 5, outputTokens: 2 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ],
  )
})
