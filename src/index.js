/**
 * DeepSeek Harness plugin registering a llama.cpp backend on the `llm` seam.
 *
 * This is an LLM adapter, not a tool: it registers a provider route with
 * `ctx.llm.registerAdapter()` and never touches `ctx.tools`.
 *
 * @module dsh-llamacpp
 */
import { LlamaCppAdapter } from './adapter.js'

export { LlamaCppAdapter, PROVIDER_HTTP_ERROR_CODE, PROVIDER_UNREACHABLE_CODE } from './adapter.js'
export { PROVIDER_ERROR_CODE, PROVIDER_PROTOCOL_ERROR_CODE, UNSUPPORTED_CONTENT_CODE } from './openai.js'

/** Where `llama-server --port 8080` listens by default, plus its OpenAI path prefix. */
export const DEFAULT_BASE_URL = 'http://127.0.0.1:8080/v1'

/** Model id sent when nothing configures one; matches a common `--alias`. */
export const DEFAULT_MODEL = 'qwen'

/** The single provider route this plugin registers unless configured otherwise. */
export const DEFAULT_PROVIDER = 'llamacpp'

/** Plugin name, as it appears in the harness plugin registry. */
export const name = 'llamacpp'

/** Hard dependency: without the `llm` seam there is nothing to register on. */
export const inject = ['llm']

/**
 * Read one non-empty string setting, preferring the patch row over the
 * environment. An empty or whitespace-only value counts as unset, because
 * that is what an exported-but-empty shell variable means in practice.
 * @param value - the configured value.
 * @param envValue - the environment fallback.
 * @param fallback - the built-in default.
 * @returns the resolved setting.
 */
function setting(value, envValue, fallback) {
  for (const candidate of [value, envValue]) {
    if (typeof candidate === 'string' && candidate.trim() !== '') return candidate.trim()
  }
  return fallback
}

/**
 * Resolve plugin configuration: patch row first, then `DSH_LLAMACPP_*`
 * environment variables, then built-in defaults.
 *
 * Note that an id-targeted cordis patch replaces the whole config object, so
 * the environment fallbacks apply to whichever fields that patch leaves out.
 * @param config - the plugin config supplied by the bundle patch or overlay.
 * @param env - environment to read; defaults to `process.env`.
 * @returns baseURL, model, optional apiKey, and the provider routes to own.
 */
export function resolveConfig(config = {}, env = process.env) {
  const provider = Array.isArray(config.provider)
    ? config.provider.map((route) => String(route).trim()).filter((route) => route !== '')
    : [DEFAULT_PROVIDER]

  return {
    baseURL: setting(config.baseURL, env.DSH_LLAMACPP_BASE_URL, DEFAULT_BASE_URL),
    model: setting(config.model, env.DSH_LLAMACPP_MODEL, DEFAULT_MODEL),
    apiKey: setting(config.apiKey, env.DSH_LLAMACPP_API_KEY, undefined),
    provider: provider.length > 0 ? provider : [DEFAULT_PROVIDER],
  }
}

/**
 * Register the llama.cpp adapter for its configured provider routes.
 * @param ctx - the harness context, with the injected `llm` seam.
 * @param config - the plugin config; see {@link resolveConfig}.
 * @returns the registration handle, released with the fiber.
 */
export function apply(ctx, config = {}) {
  const resolved = resolveConfig(config)
  const adapter = new LlamaCppAdapter(resolved)
  return ctx.llm.registerAdapter([...resolved.provider], adapter)
}
