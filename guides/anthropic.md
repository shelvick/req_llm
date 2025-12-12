# Anthropic

Access Claude models through ReqLLM's unified interface. Supports all Claude 3+ models including extended thinking.

## Configuration

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

## Provider Options

Passed via `:provider_options` keyword:

### `anthropic_top_k`
- **Type**: `1..40`
- **Purpose**: Sample from top K options per token
- **Example**: `provider_options: [anthropic_top_k: 20]`

### `anthropic_version`
- **Type**: String
- **Default**: `"2023-06-01"`
- **Purpose**: API version override
- **Example**: `provider_options: [anthropic_version: "2023-06-01"]`

### `stop_sequences`
- **Type**: List of strings
- **Purpose**: Custom stop sequences
- **Example**: `provider_options: [stop_sequences: ["END", "STOP"]]`

### `anthropic_metadata`
- **Type**: Map
- **Purpose**: Request metadata for tracking
- **Example**: `provider_options: [anthropic_metadata: %{user_id: "123"}]`

### `thinking`
- **Type**: Map
- **Purpose**: Enable extended thinking/reasoning
- **Example**: `provider_options: [thinking: %{type: "enabled", budget_tokens: 4096}]`
- **Access**: `ReqLLM.Response.thinking(response)`

### `anthropic_prompt_cache`
- **Type**: Boolean
- **Purpose**: Enable prompt caching
- **Example**: `provider_options: [anthropic_prompt_cache: true]`

### `anthropic_prompt_cache_ttl`
- **Type**: String (e.g., `"1h"`)
- **Purpose**: Cache TTL (default ~5min if omitted)
- **Example**: `provider_options: [anthropic_prompt_cache_ttl: "1h"]`

### `anthropic_cache_messages`
- **Type**: Boolean
- **Purpose**: Add cache breakpoint at last message (caches entire conversation for reuse in subsequent requests)
- **Requires**: `anthropic_prompt_cache: true`
- **Example**: `provider_options: [anthropic_prompt_cache: true, anthropic_cache_messages: true]`

> **Note**: With `anthropic_prompt_cache: true`, system messages and tools are cached by default.
> Enable `anthropic_cache_messages: true` to also cache the conversation history.
> Subsequent requests with identical messages up to the cache breakpoint will be served
> from cache, reducing costs and latency for multi-turn conversations.

## Wire Format Notes

- Endpoint: `/v1/messages`
- Auth: `x-api-key` header (not Bearer token)
- System messages: included in messages array
- Tool calls: content block structure

All differences handled automatically by ReqLLM.

## Resources

- [Anthropic API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Model Comparison](https://docs.anthropic.com/claude/docs/models-overview)
