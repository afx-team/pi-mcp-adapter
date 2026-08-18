import { describe, expect, it } from "vitest";
import { buildDeferredToolsReminder, searchDeferredTools } from "../tool-search.ts";
import type { DirectToolSpec } from "../types.ts";

const tools: DirectToolSpec[] = [
  {
    serverName: "slack",
    originalName: "send_message",
    prefixedName: "mcp__slack__send_message",
    description: "Send a message to a Slack channel",
    inputSchema: {
      type: "object",
      properties: { channel: { type: "string" }, markdown: { type: "string" } },
    },
  },
  {
    serverName: "notebook",
    originalName: "execute_cell",
    prefixedName: "mcp__notebook__execute_cell",
    description: "Execute a Jupyter notebook cell",
  },
  {
    serverName: "github",
    originalName: "search_issues",
    prefixedName: "mcp__github__search_issues",
    description: "Search repository issues",
  },
];

describe("Claude Code-compatible ToolSearch query semantics", () => {
  it("publishes the Claude Code deferred-name reminder without schemas", () => {
    expect(buildDeferredToolsReminder(["mcp__slack__send_message"]))
      .toBe(`<system-reminder>\nThe following deferred tools are now available via ToolSearch. Their schemas are NOT loaded — calling them directly will fail with InputValidationError. Use ToolSearch with query "select:<name>[,<name>...]" to load tool schemas before calling them:\nmcp__slack__send_message\n</system-reminder>`);
  });

  it("loads exact select: names in requested order", () => {
    expect(searchDeferredTools(
      tools,
      "select:mcp__github__search_issues,mcp__slack__send_message",
      5,
    ).map((match) => match.name)).toEqual([
      "mcp__github__search_issues",
      "mcp__slack__send_message",
    ]);
  });

  it("searches names, descriptions, and parameter schemas", () => {
    expect(searchDeferredTools(tools, "jupyter", 5).map((match) => match.name))
      .toEqual(["mcp__notebook__execute_cell"]);
    expect(searchDeferredTools(tools, "markdown", 5).map((match) => match.name))
      .toEqual(["mcp__slack__send_message"]);
  });

  it("supports +name requirements and ranks with remaining terms", () => {
    expect(searchDeferredTools(tools, "+slack send", 5).map((match) => match.name))
      .toEqual(["mcp__slack__send_message"]);
    expect(searchDeferredTools(tools, "+slack issues", 5)).toEqual([]);
  });

  it("returns no references for empty or unmatched searches", () => {
    expect(searchDeferredTools(tools, "", 5)).toEqual([]);
    expect(searchDeferredTools(tools, "calendar event", 5)).toEqual([]);
  });

  it("honors max_results", () => {
    expect(searchDeferredTools(tools, "search", 1)).toHaveLength(1);
  });
});
