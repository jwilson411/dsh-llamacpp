import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { LlamaCppAdapter } from '../src/adapter.js'
import { apply, inject, name, resolveConfig } from '../src/index.js'

const manifest = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'))
const patch = readFileSync(new URL('../cordis.patch.yml', import.meta.url), 'utf8')

/**
 * A context stub in the shape `apply` uses, recording every registration the
 * way the kit's tests stub `ctx.tools.register`.
 * @returns the stub context and its recorded registrations.
 */
function stubContext() {
  const registrations = []
  const handle = () => {}
  handle.replace = () => {}
  return {
    registrations,
    ctx: {
      llm: {
        registerAdapter(providers, adapter) {
          registrations.push({ providers, adapter })
          return handle
        },
      },
    },
  }
}

test('the plugin declares its name and a hard dependency on the llm seam', () => {
  assert.equal(name, 'llamacpp')
  assert.deepEqual(inject, ['llm'])
})

test('the plugin registers no tools: apply only touches ctx.llm', () => {
  const { ctx } = stubContext()
  ctx.tools = new Proxy(
    {},
    {
      get() {
        throw new assert.AssertionError({ message: 'dsh-llamacpp is an adapter, it must not touch ctx.tools' })
      },
    },
  )
  apply(ctx, {})
})

test('apply registers exactly one adapter on the llamacpp route', () => {
  const { ctx, registrations } = stubContext()
  apply(ctx, {})

  assert.equal(registrations.length, 1)
  const [{ providers, adapter }] = registrations
  assert.deepEqual(providers, ['llamacpp'])
  assert.ok(adapter instanceof LlamaCppAdapter)
  assert.equal(adapter.url, 'http://127.0.0.1:8080/v1/chat/completions')
})

test('apply returns the registration handle so the fiber can release the route', () => {
  const { ctx } = stubContext()
  const handle = apply(ctx, {})
  assert.equal(typeof handle, 'function')
  assert.equal(typeof handle.replace, 'function')
})

test('apply honours a configured provider route list', () => {
  const { ctx, registrations } = stubContext()
  apply(ctx, { provider: ['llamacpp', 'local'] })
  assert.deepEqual(registrations[0].providers, ['llamacpp', 'local'])
})

test('apply passes the resolved config through to the adapter endpoint', () => {
  const { ctx, registrations } = stubContext()
  apply(ctx, { baseURL: 'http://127.0.0.1:9999/v1', model: 'qwen3' })
  assert.equal(registrations[0].adapter.url, 'http://127.0.0.1:9999/v1/chat/completions')
})

test('resolveConfig falls back to the documented defaults', () => {
  assert.deepEqual(resolveConfig({}, {}), {
    baseURL: 'http://127.0.0.1:8080/v1',
    model: 'qwen',
    apiKey: undefined,
    provider: ['llamacpp'],
  })
})

test('resolveConfig reads the DSH_LLAMACPP_* environment fallbacks', () => {
  assert.deepEqual(
    resolveConfig(
      {},
      {
        DSH_LLAMACPP_BASE_URL: 'http://127.0.0.1:9090/v1',
        DSH_LLAMACPP_MODEL: 'qwen3-coder',
        DSH_LLAMACPP_API_KEY: 'sekret',
      },
    ),
    {
      baseURL: 'http://127.0.0.1:9090/v1',
      model: 'qwen3-coder',
      apiKey: 'sekret',
      provider: ['llamacpp'],
    },
  )
})

test('resolveConfig prefers the patch row over the environment', () => {
  const resolved = resolveConfig({ baseURL: 'http://127.0.0.1:1/v1' }, { DSH_LLAMACPP_BASE_URL: 'http://127.0.0.1:2/v1' })
  assert.equal(resolved.baseURL, 'http://127.0.0.1:1/v1')
})

test('resolveConfig treats an exported-but-empty value as unset', () => {
  assert.equal(resolveConfig({ model: '   ' }, { DSH_LLAMACPP_MODEL: '' }).model, 'qwen')
})

test('the manifest points the harness bundle at the cordis patch', () => {
  assert.equal(manifest.dsh.bundle.patch, './cordis.patch.yml')
  assert.equal(manifest.exports['./cordis.patch.yml'], './cordis.patch.yml')
  assert.ok(manifest.files.includes('cordis.patch.yml'))
  assert.ok(manifest.files.includes('README.md'))
})

test('the bundled patch inserts this package on the llamacpp id with the documented defaults', () => {
  assert.match(patch, /id:\s*llamacpp/)
  assert.match(patch, /name:\s*dsh-llamacpp/)
  assert.match(patch, /baseURL:\s*http:\/\/127\.0\.0\.1:8080\/v1/)
  assert.match(patch, /model:\s*qwen/)
})
