# dsh-llamacpp

A DeepSeek Harness **LLM adapter plugin** for a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server, spoken over its OpenAI-compatible `/v1/chat/completions` API.

**This is llama.cpp, not Ollama.** It talks to `llama-server` directly — the binary you build or
download from llama.cpp — and knows nothing about Ollama's model registry, its `/api/*` routes, or
its automatic model pulls. If you run Ollama, this is the wrong plugin.

It is also **not a tool**. It registers nothing on `ctx.tools`; it registers a provider on the
official LLM seam:

```js
ctx.llm.registerAdapter(['llamacpp'], adapter)
```

Once registered, `llamacpp` is a provider route like any other, and the harness routes model calls
to it through `ctx.llm.stream()`.

## Install

```sh
dsh plugin --profile web add github:jwilson411/dsh-llamacpp
```

The package ships `dsh.bundle.patch` pointing at [`cordis.patch.yml`](./cordis.patch.yml), so adding
it inserts a configured `llamacpp` plugin row into the profile.

## Pinned harness RC

Built against **`0.1.1-rc.2`** of the `@deepseek-ai/dsh-*` packages — the same RC as
`jwilson411/dsh-plugin-kit` and `jwilson411/dsh-spend-receipt`. Note that `@deepseek-ai/dsh-llm`'s
`latest` dist-tag still points at `0.0.1-rc.1`; the RC line lives under the `next` tag, which is why
the dev dependencies pin exact versions rather than tracking `latest`.

The seams used are `LlmAdapter`, `LlmError`, `attributionHeaders`, and `errorChain` from
`@deepseek-ai/dsh-llm`. If one of those names moves in a later RC, this package stays pinned to
`0.1.1-rc.2` until it is updated deliberately.

## Running llama.cpp

Start `llama-server` with the weights you already have. The flags that matter to this plugin:

| Flag | Why it matters here |
| --- | --- |
| `--port 8080` | The port in the default `baseURL` (`http://127.0.0.1:8080/v1`). |
| `--alias qwen` | The model **name** the server accepts and this plugin sends. Set it to match `model`, or set `model` to match it. |
| `--api-key` | If set, every request must carry `Authorization: Bearer …`. Configure `apiKey` (or `DSH_LLAMACPP_API_KEY`) to the same value. |
| `-c` / `--ctx-size` | The context window. Requests that exceed it fail at the server; this plugin surfaces that failure rather than truncating. |

Example:

```sh
llama-server -m qwen.gguf --port 8080 --alias qwen -c 32768 --api-key $DSH_LLAMACPP_API_KEY
```

Offload flags (`-ngl`, tensor split, and friends) are between you and your hardware — they change
nothing about the wire protocol, so this plugin has no opinion about them.

## Configuration

| Key | Env fallback | Default |
| --- | --- | --- |
| `baseURL` | `DSH_LLAMACPP_BASE_URL` | `http://127.0.0.1:8080/v1` |
| `model` | `DSH_LLAMACPP_MODEL` | `qwen` |
| `apiKey` | `DSH_LLAMACPP_API_KEY` | *(unset — no `Authorization` header is sent)* |
| `provider` | — | `['llamacpp']` |

The patch row wins over the environment; the environment fills in whatever the row leaves out. An
exported-but-empty variable counts as unset.

`baseURL` already includes `/v1`, and requests go to `${baseURL}/chat/completions` — the `/v1` is
never doubled. If your server sits behind a proxy that strips the prefix, set `baseURL` to the base
the proxy actually serves.

### Cordis overlay

To point the plugin at a different server, model, or credential, target the `llamacpp` id in your
profile overlay. **An id-targeted patch replaces the whole `config` object**, so repeat every field
you want, not just the one you are changing:

```yaml
- id: llamacpp
  config:
    baseURL: http://127.0.0.1:9090/v1
    model: qwen3-coder
    apiKey: change-me
```

## What it sends

Per model call, one `POST ${baseURL}/chat/completions` with `stream: true`. Every request merges
`attributionHeaders()` from `@deepseek-ai/dsh-llm` (a `User-Agent` identifying the harness — a
public product fact, no secrets) and forwards `options.signal`, so an aborted turn aborts the HTTP
request.

Request mapping:

- `options.system` is prepended as a `system` message.
- `options.messages` become OpenAI `{ role, content }` messages; text blocks are concatenated.
- `options.tools` become the OpenAI `tools` array.
- `options.temperature`, `options.stop` pass through.
- `options.maxTokens` maps to **`max_tokens`** — llama.cpp's server does not read
  `max_completion_tokens`.
- `options.model` is used when the request carries one, otherwise the configured `model`.

## What it yields

The harness `StreamChunk` protocol, in order:

1. `block-start`, index `0`, blockType `text`
2. one `text-delta` per `choices[0].delta.content` string
3. `block-end` carrying the full assembled `{ type: 'text', text }`
4. `usage`, when the server reported any (`prompt_tokens` → `inputTokens`, `completion_tokens` →
   `outputTokens`; llama.cpp has no cache split, so no cache fields are invented)
5. `finish`, last

A response that produced no text opens no block, so every `block-start` still has its `block-end`.

`finish_reason` maps to `{ kind: 'stop' }`, `{ kind: 'tool-calls' }` for `tool_calls`, and
`{ kind: 'max-tokens' }` for `length` — a truncated answer is not reported as a complete one.

Some builds and proxies ignore `stream: true` and answer with a single JSON `chat.completion`. That
is translated into exactly the same chunk sequence.

## Failures are loud

Nothing is swallowed into an empty stream. Every failure is an `LlmError` with a stable `code`:

| Code | When |
| --- | --- |
| `PROVIDER_UNREACHABLE` | Connection refused, DNS failure, reset mid-stream — usually `llama-server` is not running. |
| `PROVIDER_HTTP_ERROR` | Non-2xx answer; carries the HTTP `status` and quotes the body. |
| `PROVIDER_ERROR` | The server reported its own failure in-band, as an `error` payload. |
| `PROVIDER_PROTOCOL_ERROR` | A stream frame that is not JSON. |
| `UNSUPPORTED_CONTENT` | A message carried an image, tool-call, or tool-result block. |

A caller-driven abort is re-thrown unchanged, so the runtime reports it as an `aborted` finish
rather than a dead server.

### Text only

This is a thin adapter. It sends text and streams text back. Image blocks are **refused**, not
quietly dropped, because dropping one hands the model a conversation the caller never wrote; the
same holds for tool-call and tool-result blocks, so a full tool round-trip is out of scope here even
though `options.tools` is forwarded and a `tool_calls` finish is reported.

## Development

```sh
npm install
npm test
```

The tests mock an OpenAI-compatible server with Node's `http` and `node:test`. No GPU, no weight
download, no real llama.cpp process, and no network beyond loopback.

## License

MIT © 2026 jwilson411
