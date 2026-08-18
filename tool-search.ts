import type { DirectToolSpec } from "./types.ts";

export const TOOL_SEARCH_NAME = "ToolSearch";
export const DEFAULT_TOOL_SEARCH_RESULTS = 5;

export const TOOL_SEARCH_DESCRIPTION = `Fetches full schema definitions for deferred tools so they can be called.

Deferred tools appear by name in <system-reminder> messages. Until fetched, only the name is known — there is no parameter schema, so the tool cannot be invoked. This tool takes a query, matches it against the deferred tool list, and returns the matched tools' complete JSONSchema definitions inside a <functions> block. Once a tool's schema appears in that result, it is callable exactly like any tool defined at the top of the prompt.

Result format: each matched tool appears as one <function>{"description": "...", "name": "...", "parameters": {...}}</function> line inside the <functions> block — the same encoding as the tool list at the top of this prompt.

Query forms:
- "select:Read,Edit,Grep" — fetch these exact tools by name
- "notebook jupyter" — keyword search, up to max_results best matches
- "+slack send" — require "slack" in the name, rank by remaining terms`;

export function buildDeferredToolsReminder(toolNames: readonly string[]): string {
  return `<system-reminder>\nThe following deferred tools are now available via ToolSearch. Their schemas are NOT loaded — calling them directly will fail with InputValidationError. Use ToolSearch with query "select:<name>[,<name>...]" to load tool schemas before calling them:\n${toolNames.join("\n")}\n</system-reminder>`;
}

export interface ToolSearchMatch {
  name: string;
  server: string;
  description: string;
}

function normalize(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_./:-]+/g, " ")
    .toLowerCase();
}

function tokens(value: string): string[] {
  return normalize(value).split(/[^a-z0-9]+/).filter(Boolean);
}

function searchableText(spec: DirectToolSpec): string {
  let schema = "";
  try {
    schema = spec.inputSchema === undefined ? "" : JSON.stringify(spec.inputSchema);
  } catch {
    // A malformed metadata cache must not make tool discovery fail.
  }
  return normalize(`${spec.prefixedName} ${spec.originalName} ${spec.serverName} ${spec.description} ${schema}`);
}

function score(spec: DirectToolSpec, queryTokens: string[]): number | null {
  const name = normalize(spec.prefixedName);
  const originalName = normalize(spec.originalName);
  const description = normalize(spec.description);
  const haystack = searchableText(spec);
  let matched = 0;
  let value = 0;

  for (const token of queryTokens) {
    if (!haystack.includes(token)) continue;
    matched += 1;
    if (name === token || originalName === token) value += 80;
    else if (name.startsWith(token) || originalName.startsWith(token)) value += 45;
    else if (name.includes(token) || originalName.includes(token)) value += 30;
    else if (description.includes(token)) value += 12;
    else value += 5;
  }

  if (matched === 0 || matched !== queryTokens.length) return null;
  return value + Math.round((matched / queryTokens.length) * 20);
}

/** Resolve the same query forms exposed by Claude Code's client-side ToolSearch. */
export function searchDeferredTools(
  specs: readonly DirectToolSpec[],
  query: string,
  maxResults = DEFAULT_TOOL_SEARCH_RESULTS,
): ToolSearchMatch[] {
  const limit = Number.isFinite(maxResults)
    ? Math.max(1, Math.min(100, Math.trunc(maxResults)))
    : DEFAULT_TOOL_SEARCH_RESULTS;
  const trimmed = query.trim();
  if (!trimmed) return [];

  if (trimmed.toLowerCase().startsWith("select:")) {
    const requested = trimmed.slice(trimmed.indexOf(":") + 1)
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean);
    const byName = new Map(specs.map((spec) => [spec.prefixedName, spec]));
    return requested
      .map((name) => byName.get(name))
      .filter((spec): spec is DirectToolSpec => spec !== undefined)
      .slice(0, limit)
      .map(toMatch);
  }

  const requiredNameTerms: string[] = [];
  const rankingTerms: string[] = [];
  for (const term of trimmed.split(/\s+/).filter(Boolean)) {
    if (term.startsWith("+") && term.length > 1) requiredNameTerms.push(...tokens(term.slice(1)));
    else rankingTerms.push(...tokens(term));
  }
  const effectiveRankingTerms = rankingTerms.length > 0 ? rankingTerms : requiredNameTerms;

  return specs
    .filter((spec) => {
      const name = normalize(spec.prefixedName);
      return requiredNameTerms.every((term) => name.includes(term));
    })
    .map((spec) => ({ spec, score: score(spec, effectiveRankingTerms) }))
    .filter((entry): entry is { spec: DirectToolSpec; score: number } => entry.score !== null)
    .sort((a, b) => b.score - a.score || a.spec.prefixedName.localeCompare(b.spec.prefixedName))
    .slice(0, limit)
    .map(({ spec }) => toMatch(spec));
}

function toMatch(spec: DirectToolSpec): ToolSearchMatch {
  return {
    name: spec.prefixedName,
    server: spec.serverName,
    description: spec.description,
  };
}
