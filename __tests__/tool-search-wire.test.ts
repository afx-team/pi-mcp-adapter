import { describe, expect, it } from "vitest";
import { stream } from "@earendil-works/pi-ai/api/anthropic-messages";

describe("ToolSearch Anthropic wire protocol", () => {
  it("serializes additive activation as tool_reference plus deferred definitions", async () => {
    let payload: any;
    const model = {
      id: "claude-sonnet-4-5-20250929",
      name: "Claude Sonnet 4.5",
      api: "anthropic-messages",
      provider: "anthropic",
      baseUrl: "https://api.anthropic.com",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 200_000,
      maxTokens: 8_192,
      compat: { supportsToolReferences: true },
    } as any;
    const usage = {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    };
    const context = {
      messages: [
        { role: "user", content: "Open the site", timestamp: 1 },
        {
          role: "assistant",
          content: [{
            type: "toolCall",
            id: "toolu_search",
            name: "ToolSearch",
            arguments: { query: "select:mcp__browser__navigate", max_results: 5 },
          }],
          api: "anthropic-messages",
          provider: "anthropic",
          model: model.id,
          usage,
          stopReason: "toolUse",
          timestamp: 2,
        },
        {
          role: "toolResult",
          toolCallId: "toolu_search",
          toolName: "ToolSearch",
          content: [{ type: "text", text: "Loaded tools: mcp__browser__navigate" }],
          addedToolNames: ["mcp__browser__navigate"],
          isError: false,
          timestamp: 3,
        },
      ],
      tools: [
        {
          name: "ToolSearch",
          description: "Fetches full schema definitions for deferred tools so they can be called.",
          parameters: {
            type: "object",
            properties: { query: { type: "string" }, max_results: { type: "number" } },
            required: ["query", "max_results"],
          },
        },
        {
          name: "mcp__browser__navigate",
          description: "Navigate the browser",
          parameters: {
            type: "object",
            properties: { url: { type: "string" } },
            required: ["url"],
          },
        },
      ],
    } as any;

    const events = stream(model, context, {
      client: {} as any,
      cacheRetention: "none",
      onPayload(nextPayload) {
        payload = nextPayload;
        throw new Error("payload captured");
      },
    });
    for await (const _event of events) {
      // The deliberate onPayload error stops before network I/O.
    }

    expect(payload.tools).toEqual([
      expect.objectContaining({ name: "ToolSearch" }),
      expect.objectContaining({
        name: "mcp__browser__navigate",
        defer_loading: true,
      }),
    ]);
    const searchResult = payload.messages.at(-1).content[0];
    expect(searchResult).toEqual({
      type: "tool_result",
      tool_use_id: "toolu_search",
      content: [{ type: "tool_reference", tool_name: "mcp__browser__navigate" }],
      is_error: false,
    });
  });
});
