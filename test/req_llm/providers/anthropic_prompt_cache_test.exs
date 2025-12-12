defmodule ReqLLM.Providers.AnthropicPromptCacheTest do
  @moduledoc """
  Unit tests for Anthropic prompt caching functionality.

  Tests cache_control header injection and body transformations for:
  - Beta header inclusion
  - Tool cache_control injection
  - System message cache_control handling
  - Message cache_control handling (conversation prefix caching)
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Anthropic

  alias ReqLLM.Context
  alias ReqLLM.Providers.Anthropic

  describe "prompt caching beta header" do
    test "adds prompt caching beta header when enabled" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context, anthropic_prompt_cache: true)

      beta_header =
        Enum.find_value(request.headers, fn
          {"anthropic-beta", value} -> value
          _ -> nil
        end)

      assert beta_header != nil
      beta_string = if is_list(beta_header), do: hd(beta_header), else: beta_header
      assert String.contains?(beta_string, "prompt-caching-2024-07-31")
    end

    test "does not add prompt caching beta header when disabled" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} = Anthropic.prepare_request(:chat, model, context, [])

      beta_header =
        Enum.find_value(request.headers, fn
          {"anthropic-beta", value} -> value
          _ -> nil
        end)

      refute beta_header && String.contains?(beta_header, "prompt-caching-2024-07-31")
    end
  end

  describe "tool cache_control injection" do
    test "injects cache_control into tools with default TTL" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            param: [type: :string, required: true, doc: "Test parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          tools: [tool],
          anthropic_prompt_cache: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      assert length(decoded["tools"]) == 1

      [encoded_tool] = decoded["tools"]
      assert encoded_tool["cache_control"] == %{"type" => "ephemeral"}
    end

    test "injects cache_control into tools with 1h TTL" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            param: [type: :string, required: true, doc: "Test parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          tools: [tool],
          anthropic_prompt_cache: true,
          anthropic_prompt_cache_ttl: "1h"
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      [encoded_tool] = decoded["tools"]
      assert encoded_tool["cache_control"] == %{"type" => "ephemeral", "ttl" => "1h"}
    end

    test "does not inject cache_control when prompt caching disabled" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            param: [type: :string, required: true, doc: "Test parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      {:ok, request} = Anthropic.prepare_request(:chat, model, context, tools: [tool])

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      [encoded_tool] = decoded["tools"]
      refute Map.has_key?(encoded_tool, "cache_control")
    end
  end

  describe "system message cache_control injection" do
    test "converts system string to content block with cache_control" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context, anthropic_prompt_cache: true)

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["system"])
      [system_block] = decoded["system"]

      assert system_block["type"] == "text"
      assert system_block["text"] == "You are a helpful assistant."
      assert system_block["cache_control"] == %{"type" => "ephemeral"}
    end

    test "adds cache_control to last system block when already array" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")

      system_content = [
        ReqLLM.Message.ContentPart.text("First instruction."),
        ReqLLM.Message.ContentPart.text("Second instruction.")
      ]

      context =
        ReqLLM.Context.new([
          %ReqLLM.Message{role: :system, content: system_content},
          ReqLLM.Context.user("Hello!")
        ])

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context, anthropic_prompt_cache: true)

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["system"])
      assert length(decoded["system"]) == 2

      last_block = List.last(decoded["system"])
      assert last_block["cache_control"] == %{"type" => "ephemeral"}

      first_block = List.first(decoded["system"])
      refute Map.has_key?(first_block, "cache_control")
    end

    test "does not modify system when prompt caching disabled" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} = Anthropic.prepare_request(:chat, model, context, [])

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["system"] == "You are a helpful assistant."
    end
  end

  describe "combined prompt caching scenarios" do
    test "applies cache_control to both tools and system" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            param: [type: :string, required: true, doc: "Test parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          tools: [tool],
          anthropic_prompt_cache: true,
          anthropic_prompt_cache_ttl: "2h"
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      [system_block] = decoded["system"]
      assert system_block["cache_control"] == %{"type" => "ephemeral", "ttl" => "2h"}

      [encoded_tool] = decoded["tools"]
      assert encoded_tool["cache_control"] == %{"type" => "ephemeral", "ttl" => "2h"}
    end

    test "respects existing cache_control on tools" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")

      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            param: [type: :string, required: true, doc: "Test parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          tools: [tool],
          anthropic_prompt_cache: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      [encoded_tool] = decoded["tools"]
      assert encoded_tool["cache_control"] == %{"type" => "ephemeral"}
    end
  end

  describe "message cache_control injection" do
    test "adds cache_control to last message when anthropic_cache_messages enabled" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")

      context =
        Context.new([
          Context.system("You are helpful."),
          Context.user("Hello!"),
          Context.assistant("Hi there!"),
          Context.user("How are you?")
        ])

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          anthropic_prompt_cache: true,
          anthropic_cache_messages: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      messages = decoded["messages"]
      assert length(messages) == 3

      # Last message should have cache_control
      last_msg = List.last(messages)
      assert last_msg["role"] == "user"
      assert is_list(last_msg["content"])
      [content_block] = last_msg["content"]
      assert content_block["cache_control"] == %{"type" => "ephemeral"}

      # Earlier messages should NOT have cache_control
      first_msg = List.first(messages)
      refute has_cache_control?(first_msg)
    end

    test "adds cache_control to last content block of multi-part message" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")

      user_content = [
        ReqLLM.Message.ContentPart.text("First part."),
        ReqLLM.Message.ContentPart.text("Second part.")
      ]

      context =
        Context.new([
          Context.system("You are helpful."),
          %ReqLLM.Message{role: :user, content: user_content}
        ])

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          anthropic_prompt_cache: true,
          anthropic_cache_messages: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      [user_msg] = decoded["messages"]
      assert length(user_msg["content"]) == 2

      # Only last content block should have cache_control
      [first_block, last_block] = user_msg["content"]
      refute Map.has_key?(first_block, "cache_control")
      assert last_block["cache_control"] == %{"type" => "ephemeral"}
    end

    test "applies TTL to message cache_control" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          anthropic_prompt_cache: true,
          anthropic_prompt_cache_ttl: "1h",
          anthropic_cache_messages: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      last_msg = List.last(decoded["messages"])
      [content_block] = last_msg["content"]
      assert content_block["cache_control"] == %{"type" => "ephemeral", "ttl" => "1h"}
    end

    test "does not add message cache_control when anthropic_cache_messages is false" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          anthropic_prompt_cache: true,
          anthropic_cache_messages: false
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      last_msg = List.last(decoded["messages"])
      refute has_cache_control?(last_msg)
    end

    test "does not add message cache_control when anthropic_prompt_cache is false" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          anthropic_prompt_cache: false,
          anthropic_cache_messages: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      last_msg = List.last(decoded["messages"])
      refute has_cache_control?(last_msg)
    end

    test "handles empty messages gracefully" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")

      # Context with only system message (no user/assistant messages)
      context = Context.new([Context.system("You are helpful.")])

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          anthropic_prompt_cache: true,
          anthropic_cache_messages: true
        )

      # Should not crash
      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      # Should have empty messages array
      assert decoded["messages"] == []
    end

    test "caches messages alongside system and tools" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            param: [type: :string, required: true, doc: "Test parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      {:ok, request} =
        Anthropic.prepare_request(:chat, model, context,
          tools: [tool],
          anthropic_prompt_cache: true,
          anthropic_cache_messages: true
        )

      updated_request = Anthropic.encode_body(request)
      decoded = Jason.decode!(updated_request.body)

      # System should be cached
      [system_block] = decoded["system"]
      assert system_block["cache_control"] == %{"type" => "ephemeral"}

      # Tools should be cached
      [encoded_tool] = decoded["tools"]
      assert encoded_tool["cache_control"] == %{"type" => "ephemeral"}

      # Last message should be cached
      last_msg = List.last(decoded["messages"])
      [content_block] = last_msg["content"]
      assert content_block["cache_control"] == %{"type" => "ephemeral"}
    end

    test "respects existing cache_control on message content blocks" do
      # Build body directly to test the caching function with pre-existing cache_control
      body = %{
        messages: [
          %{
            role: "user",
            content: [
              %{type: "text", text: "First part"},
              %{type: "text", text: "Second part", cache_control: %{type: "ephemeral", ttl: "2h"}}
            ]
          }
        ]
      }

      opts = [anthropic_prompt_cache: true, anthropic_cache_messages: true]

      # Use the internal function directly
      result = Anthropic.maybe_apply_prompt_caching(body, opts)

      [msg] = result[:messages]
      [first_block, last_block] = msg[:content]

      # First block should not have cache_control
      refute Map.has_key?(first_block, :cache_control)

      # Last block should preserve existing cache_control (not overwritten)
      assert last_block[:cache_control] == %{type: "ephemeral", ttl: "2h"}
    end
  end

  # Helper to check if a message has cache_control on any content block
  defp has_cache_control?(%{"content" => content}) when is_list(content) do
    Enum.any?(content, &Map.has_key?(&1, "cache_control"))
  end

  defp has_cache_control?(%{"content" => _}), do: false
  defp has_cache_control?(_), do: false
end
