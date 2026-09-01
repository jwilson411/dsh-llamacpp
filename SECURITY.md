# Security Policy

## Reporting a Vulnerability

Please do not open a public GitHub issue for a security report.

Use GitHub's private advisory form:

https://github.com/jwilson411/dsh-llamacpp/security/advisories/new

Include the version or commit, steps to reproduce, and what an attacker gains.

## Scope

dsh-llamacpp is a DeepSeek Harness LLM adapter plugin. It is not a tool. It registers a provider on `ctx.llm.registerAdapter` and posts one HTTP request per model call to a local llama.cpp `llama-server` over that server's OpenAI-compatible `/v1/chat/completions` API.

This is llama.cpp, not Ollama. The plugin does not talk to Ollama's model registry, `/api/*` routes, or automatic model pulls.

The default `baseURL` is `http://127.0.0.1:8080/v1`. An optional `apiKey` (plugin config or `DSH_LLAMACPP_API_KEY`) is sent as `Authorization: Bearer` when set. The plugin does not log that key. Error messages quote the HTTP status and a truncated response body from the server, not the credential.

Pointing `baseURL` at a remote host is a deployment choice. An attacker who already controls the process running the harness, or who can reach a llama-server you exposed beyond loopback without a network boundary, is out of scope.

## Supported versions

Only the latest release receives security fixes.
