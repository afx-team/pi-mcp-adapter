import type { AgentToolUpdateCallback, ExtensionAPI, ExtensionContext, ToolInfo } from "@earendil-works/pi-coding-agent";
import type { McpExtensionState } from "./state.ts";
import type { DirectToolSpec, McpAdapterOptions, McpConfig, PromptMetadata } from "./types.ts";
import type { McpOAuthRuntime } from "./mcp-auth-flow.ts";
import { Type } from "typebox";
import type { TSchema } from "typebox";
import { showStatus, showTools, showPrompts, reconnectServer, reconnectServers, authenticateServer, logoutServer, openMcpAuthPanel, openMcpPanel, openMcpSetup } from "./commands.ts";
import { cloneMcpConfig, loadMcpConfig, writeProjectServerDisabledOverride } from "./config.ts";
import { buildProxyDescription, createDirectToolExecutor, getMissingConfiguredDirectToolServers, resolveAllMcpTools, resolveDirectTools } from "./direct-tools.ts";
import { flushMetadataCache, initializeMcp, updateStatusBar } from "./init.ts";
import { loadMetadataCache, type MetadataCache } from "./metadata-cache.ts";
import { createPromptCommand, resolveCachedPrompts } from "./prompts.ts";
import { logger } from "./logger.ts";
import { executeAuthComplete, executeAuthStart, executeCall, executeConnect, executeDescribe, executeInstructions, executeList, executeSearch, executeStatus, executeUiMessages } from "./proxy-modes.ts";
import { formatTerminalError, getConfigPathFromArgv, normalizeDirectToolInputSchema, truncateAtWord } from "./utils.ts";
import { createOAuthRuntime, shutdownOAuth } from "./mcp-auth-flow.ts";
import { createMcpDirectToolCallRenderer, createMcpProxyToolCallRenderer, createMcpToolResultRenderer, resolveMcpToolRenderOptions } from "./tool-result-renderer.ts";
import { toolErrorOverride } from "./error-signal.ts";
import { createMcpRuntimeOwner, createOwnedUi, isAbortError, type McpRuntimeOwner } from "./runtime-owner.ts";
import { publishMcpStatusShutdown } from "./mcp-status.ts";
import { runMcpScript } from "./mcp-code.ts";
import { cleanupMaterializedBinaryResources } from "./tool-registrar.ts";
import { buildDeferredToolsReminder, DEFAULT_TOOL_SEARCH_RESULTS, searchDeferredTools, TOOL_SEARCH_DESCRIPTION, TOOL_SEARCH_NAME } from "./tool-search.ts";

export type { McpAdapterOptions } from "./types.ts";
export {
  MCP_STATUS_EVENT,
  MCP_STATUS_SNAPSHOT_VERSION,
  MCP_TOOL_APPROVAL_REQUEST_EVENT,
  type McpServerRuntimeStatus,
  type McpServerStatusSnapshot,
  type McpStatusSnapshot,
  type McpToolApprovalDecision,
  type McpToolApprovalHandler,
  type McpToolApprovalOrigin,
  type McpToolApprovalRequest,
} from "./types.ts";

const INIT_WAIT_TIMEOUT_MS = 30_000;
const INIT_WAIT_TIMED_OUT: unique symbol = Symbol("init-wait-timed-out");

async function awaitWithTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T | typeof INIT_WAIT_TIMED_OUT> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<typeof INIT_WAIT_TIMED_OUT>((resolve) => {
        timer = setTimeout(() => resolve(INIT_WAIT_TIMED_OUT), timeoutMs);
        timer.unref?.();
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

// TypeBox 1.x annotates raw objects passed to Type.Optional with an enumerable
// "~optional" key that survives serialization into provider tool schemas (Gemini
// rejects it with 400 INVALID_ARGUMENT). Prefer a real Type.Number schema; fall
// back to a plain raw schema for host TypeBox shims that omit Type.Number, since
// a property left out of `required` is optional by default.
function optionalNumber(options: { minimum?: number; description: string }): TSchema {
  const number = (Type as { Number?: (opts: typeof options) => TSchema }).Number;
  return typeof number === "function"
    ? Type.Optional(number(options))
    : ({ type: "number", ...options } as unknown as TSchema);
}

function requiredNumber(options: { minimum?: number; maximum?: number; default?: number; description: string }): TSchema {
  const number = (Type as { Number?: (opts: typeof options) => TSchema }).Number;
  return typeof number === "function"
    ? number(options)
    : ({ type: "number", ...options } as unknown as TSchema);
}

function installMcpAdapter(pi: ExtensionAPI, options: McpAdapterOptions) {
  const sessionConfig = options.config !== undefined ? cloneMcpConfig(options.config) : undefined;
  const programmaticConfig = sessionConfig !== undefined;
  let state: McpExtensionState | null = null;
  let initPromise: Promise<McpExtensionState> | null = null;
  let currentOwner: McpRuntimeOwner | null = null;
  let currentOAuthRuntime: McpOAuthRuntime | null = null;
  let lifecycleGeneration = 0;

  async function shutdownState(currentState: McpExtensionState | null, reason: string): Promise<void> {
    if (!currentState) {
      publishMcpStatusShutdown(pi.events);
      return;
    }

    publishMcpStatusShutdown(currentState.statusEvents);

    if (currentState.uiServer) {
      currentState.uiServer.close(reason);
      currentState.uiServer = null;
    }

    let flushError: unknown;
    try {
      flushMetadataCache(currentState);
    } catch (error) {
      flushError = error;
    }

    try {
      if (currentState.owner) {
        await currentState.owner.stop(reason);
      } else {
        await currentState.lifecycle.gracefulShutdown();
      }
    } catch (error) {
      if (flushError) {
        console.error(`MCP: graceful shutdown failed after metadata flush error: ${formatTerminalError(error)}`);
      } else {
        throw error;
      }
    }

    if (flushError) {
      throw flushError;
    }
  }

  const earlyConfigPath = programmaticConfig
    ? undefined
    : options.configPath ?? getConfigPathFromArgv();
  const earlyConfig = programmaticConfig
    ? cloneMcpConfig(sessionConfig)
    : loadMcpConfig(earlyConfigPath);
  const earlyCache = loadMetadataCache();
  const envRaw = process.env.MCP_DIRECT_TOOLS;
  const envDirectToolOverride = envRaw?.split(",").map(s => s.trim()).filter(Boolean);
  const registeredDirectTools = new Map<string, string>();
  const registeredDeferredTools = new Map<string, string>();
  const deferredToolCatalog = new Map<string, DirectToolSpec>();
  const loadedDeferredTools = new Set<string>();
  let lastDeferredReminderFingerprint: string | undefined;
  const fallbackDeactivatedTools = new Set<string>();
  const toolRenderOptions = resolveMcpToolRenderOptions(earlyConfig.settings);
  const toolRenderShell = toolRenderOptions.resultRendering === "compact" ? "self" : "default";
  const renderMcpToolResult = createMcpToolResultRenderer(toolRenderOptions);
  let proxyToolRegistered = false;
  let proxyToolDescription: string | null = null;
  let directToolsFrozen = false;

  // OMP remaps `typebox` to a host shim that historically lacked Type.Unsafe.
  // Prefer Unsafe when present (real TypeBox / fixed OMP shim); otherwise pass
  // the normalized JSON Schema through as a plain object so toolWireSchema and
  // validateToolArguments still treat it as JSON Schema.
  const toToolParameters = (schema: Record<string, unknown>) =>
    typeof (Type as { Unsafe?: (value: never) => unknown }).Unsafe === "function"
      ? (Type as { Unsafe: (value: never) => unknown }).Unsafe(schema as never)
      : schema;

  function directToolFingerprint(spec: DirectToolSpec): string {
    return JSON.stringify({
      serverName: spec.serverName,
      originalName: spec.originalName,
      prefixedName: spec.prefixedName,
      description: spec.description,
      inputSchema: spec.inputSchema,
      resourceUri: spec.resourceUri,
      uiResourceUri: spec.uiResourceUri,
      uiStreamMode: spec.uiStreamMode,
    });
  }

  function registerMcpTool(spec: DirectToolSpec, alwaysLoad: boolean): void {
    (pi.registerTool as (tool: unknown) => unknown)({
      name: spec.prefixedName,
      label: `MCP: ${spec.originalName}`,
      description: spec.description || "(no description)",
      ...(alwaysLoad
        ? { promptSnippet: truncateAtWord(spec.description, 100) || `MCP tool from ${spec.serverName}` }
        : {}),
      parameters: toToolParameters(normalizeDirectToolInputSchema(spec.inputSchema)),
      execute: createDirectToolExecutor(() => state, () => initPromise, spec),
      renderShell: toolRenderShell,
      renderCall: createMcpDirectToolCallRenderer(spec.prefixedName, toolRenderOptions),
      renderResult: renderMcpToolResult,
    });
  }

  function registerDirectTool(spec: DirectToolSpec): void {
    registerMcpTool(spec, true);
  }

  function resolveCurrentDirectTools(config: McpConfig, cache: MetadataCache | null): DirectToolSpec[] {
    if (envRaw === "__none__") return [];
    const prefix = config.settings?.toolPrefix ?? "server";
    return resolveDirectTools(config, cache, prefix, envDirectToolOverride);
  }

  function getActiveToolsIfReady(): string[] | undefined {
    try {
      return pi.getActiveTools?.();
    } catch (error) {
      if (error instanceof Error
        && error.message.includes("Action methods cannot be called during extension loading")) return undefined;
      throw error;
    }
  }

  function deactivateTools(toolNames: string[]): string[] {
    if (toolNames.length === 0) return [];
    const unregisterTool = (pi as ExtensionAPI & { unregisterTool?: (name: string) => boolean }).unregisterTool;
    const unregistered = toolNames.filter((toolName) => unregisterTool?.(toolName) === true);
    const fallbackNames = toolNames.filter((toolName) => !unregistered.includes(toolName));
    const remove = new Set(toolNames);
    const activeTools = getActiveToolsIfReady();
    if (!activeTools || activeTools.length === 0) {
      for (const toolName of fallbackNames) fallbackDeactivatedTools.add(toolName);
      return unregistered;
    }
    const nextActiveTools = activeTools.filter((name) => !remove.has(name));
    if (nextActiveTools.length !== activeTools.length) {
      for (const toolName of fallbackNames) fallbackDeactivatedTools.add(toolName);
      pi.setActiveTools(nextActiveTools);
    }
    return unregistered;
  }

  function removeToolsFromActiveSet(toolNames: Iterable<string>): void {
    const remove = new Set(toolNames);
    if (remove.size === 0) return;
    const activeTools = getActiveToolsIfReady();
    if (!activeTools) return;
    const nextActiveTools = activeTools.filter((name) => !remove.has(name));
    if (nextActiveTools.length !== activeTools.length) pi.setActiveTools(nextActiveTools);
  }

  function syncDirectTools(config: McpConfig, cache: MetadataCache | null): {
    specs: DirectToolSpec[];
    added: string[];
    updated: string[];
    deactivated: string[];
  } {
    const specs = resolveCurrentDirectTools(config, cache);
    const nextNames = new Set(specs.map((spec) => spec.prefixedName));
    const added: string[] = [];
    const updated: string[] = [];
    const deactivated: string[] = [];

    for (const spec of specs) {
      const fingerprint = directToolFingerprint(spec);
      const previous = registeredDirectTools.get(spec.prefixedName);
      if (previous !== fingerprint) {
        registerDirectTool(spec);
        registeredDirectTools.set(spec.prefixedName, fingerprint);
        if (fallbackDeactivatedTools.delete(spec.prefixedName)) {
          const activeTools = getActiveToolsIfReady();
          if (activeTools && !activeTools.includes(spec.prefixedName)) {
            pi.setActiveTools([...activeTools, spec.prefixedName]);
          }
        }
        (previous ? updated : added).push(spec.prefixedName);
      }
    }

    for (const toolName of [...registeredDirectTools.keys()]) {
      if (nextNames.has(toolName)) continue;
      registeredDirectTools.delete(toolName);
      deactivated.push(toolName);
    }

    deactivateTools(deactivated);
    return { specs, added, updated, deactivated };
  }

  function syncDeferredTools(config: McpConfig, cache: MetadataCache | null, directSpecs: DirectToolSpec[]): {
    added: string[];
    updated: string[];
    deactivated: string[];
  } {
    const directIdentities = new Set(directSpecs.map((spec) => `${spec.serverName}\0${spec.originalName}`));
    const specs = config.settings?.disableToolSearch === true
      ? []
      : resolveAllMcpTools(config, cache, "claude")
        .filter((spec) => !directIdentities.has(`${spec.serverName}\0${spec.originalName}`));
    const nextNames = new Set(specs.map((spec) => spec.prefixedName));
    const added: string[] = [];
    const updated: string[] = [];
    const deactivated: string[] = [];

    deferredToolCatalog.clear();
    for (const spec of specs) {
      deferredToolCatalog.set(spec.prefixedName, spec);
      const fingerprint = directToolFingerprint(spec);
      const previous = registeredDeferredTools.get(spec.prefixedName);
      if (previous !== fingerprint) {
        registerMcpTool(spec, false);
        registeredDeferredTools.set(spec.prefixedName, fingerprint);
        (previous ? updated : added).push(spec.prefixedName);
      }
    }

    for (const toolName of [...registeredDeferredTools.keys()]) {
      if (nextNames.has(toolName)) continue;
      registeredDeferredTools.delete(toolName);
      loadedDeferredTools.delete(toolName);
      deactivated.push(toolName);
    }

    // registerTool() makes a new runtime tool active. Deferred tools remain
    // inactive until ToolSearch loads them; previously loaded tools stay active.
    removeToolsFromActiveSet(specs
      .map((spec) => spec.prefixedName)
      .filter((name) => !loadedDeferredTools.has(name)));
    // A tool can move from deferred to always-loaded with the same Claude-style
    // name. In that case registerDirectTool() already replaced its definition;
    // do not unregister the newly installed direct tool.
    deactivateTools(deactivated.filter((name) => !registeredDirectTools.has(name)));
    if (added.length + updated.length + deactivated.length > 0) {
      lastDeferredReminderFingerprint = undefined;
    }
    return { added, updated, deactivated };
  }

  function applyDirectToolConfigChanges(changes: Map<string, true | string[] | false>): void {
    if (!state) return;
    for (const [serverName, value] of changes) {
      const definition = state.config.mcpServers[serverName];
      if (!definition) continue;
      state.config.mcpServers[serverName] = { ...definition, directTools: value };
    }
  }

  function syncToolSurface(ctx?: ExtensionContext): void {
    const config = state?.config ?? earlyConfig;
    const cache = loadMetadataCache();
    const result = syncDirectTools(config, cache);
    const deferredResult = syncDeferredTools(config, cache, result.specs);
    syncProxyTool(config, cache, result.specs);
    const changed = result.added.length + result.updated.length + result.deactivated.length
      + deferredResult.added.length + deferredResult.updated.length + deferredResult.deactivated.length;
    if (changed > 0 && ctx?.hasUI) {
      ctx.ui.notify(
        `MCP: tool surface refreshed (+${result.added.length + deferredResult.added.length}, ~${result.updated.length + deferredResult.updated.length}, -${result.deactivated.length + deferredResult.deactivated.length})`,
        "info",
      );
    }
  }

  const registeredPromptCommands = new Set<string>();

  function registerPromptCommands(specs: Iterable<PromptMetadata>): void {
    for (const spec of specs) {
      if (registeredPromptCommands.has(spec.commandName)) {
        logger.debug(`MCP: prompt "${spec.originalName}" on ${spec.serverName} skipped; /${spec.commandName} is already registered`);
        continue;
      }
      registeredPromptCommands.add(spec.commandName);
      pi.registerCommand(spec.commandName, createPromptCommand(pi, () => state, spec));
    }
  }

  function syncPromptCommands(): void {
    registerPromptCommands([...(state?.promptMetadata?.values() ?? [])].flat());
  }

  registerPromptCommands(resolveCachedPrompts(earlyConfig));

  const getPiTools = (): ToolInfo[] => pi.getAllTools();

  pi.registerFlag("mcp-config", {
    description: "Path to MCP config file",
    type: "string",
  });

  function startInitialization(ctx: ExtensionContext, owner: McpRuntimeOwner, oauthRuntime: McpOAuthRuntime, generation: number, staleReason: string): Promise<void> {
    owner.addCleanup(() => cleanupMaterializedBinaryResources(owner.signal));
    const promise = initializeMcp(pi, ctx, owner, {
      ...(programmaticConfig || options.configPath !== undefined
        ? {
            ...(earlyConfigPath !== undefined ? { configPath: earlyConfigPath } : {}),
            ...(sessionConfig !== undefined ? { config: sessionConfig } : {}),
          }
        : {}),
      oauthRuntime,
      statusEvents: pi.events,
    });
    initPromise = promise;

    return promise.then(async (nextState) => {
      if (!owner.isActive() || generation !== lifecycleGeneration || initPromise !== promise) {
        try {
          await shutdownState(nextState, staleReason);
        } catch (error) {
          console.error(`MCP: failed to clean stale initialization state: ${formatTerminalError(error)}`);
        }
        return;
      }

      state = nextState;
      nextState.onToolMetadataUpdated = (_serverName, _reason) => {
        if (state !== nextState || !owner.isActive()) return;
        syncPromptCommands();
        if (directToolsFrozen) {
          const cache = loadMetadataCache();
          syncDeferredTools(nextState.config, cache, resolveCurrentDirectTools(nextState.config, cache));
          logger.debug(`MCP: metadata update for ${_serverName} (${_reason}) refreshed deferred catalog — directTools frozen`);
          return;
        }
        syncToolSurface(ctx);
      };
      syncPromptCommands();
      syncToolSurface(ctx);
      updateStatusBar(nextState);
      initPromise = null;
      if (earlyConfig.settings?.freezeDirectTools === true) {
        directToolsFrozen = true;
        logger.info("MCP: direct tools frozen after initial sync — reconnects won't rebuild the system prompt; use mcp({ connect: \"server\" }) to rediscover");
      }
    }).catch(async err => {
      if (!owner.isActive() || generation !== lifecycleGeneration) {
        return;
      }
      if (initPromise !== promise && initPromise !== null) {
        return;
      }
      console.error(`MCP initialization failed: ${formatTerminalError(err)}`);
      initPromise = null;
      if (state) return;

      try {
        await Promise.all([
          owner.stop("MCP initialization failed"),
          shutdownOAuth(oauthRuntime),
        ]);
      } catch (error) {
        console.error(`MCP: failed to clean rejected initialization: ${formatTerminalError(error)}`);
      }
    });
  }

  function startLoadTimeInitialization(): void {
    const hasStartupServer = Object.values(earlyConfig.mcpServers).some((definition) => {
      if (definition.disabled === true) return false;
      return definition.lifecycle === "eager" || definition.lifecycle === "keep-alive";
    });
    if (!hasStartupServer) return;
    setImmediate(() => {
      if (lifecycleGeneration !== 0 || state || initPromise) return;
      const generation = ++lifecycleGeneration;
      const owner = createMcpRuntimeOwner();
      const oauthRuntime = createOAuthRuntime(owner.signal);
      currentOwner = owner;
      currentOAuthRuntime = oauthRuntime;
      startInitialization({
        mode: "print",
        hasUI: false,
        cwd: process.cwd(),
        model: undefined,
        modelRegistry: undefined,
        signal: undefined,
      } as unknown as ExtensionContext, owner, oauthRuntime, generation, "stale_load_time_initialization");
    });
  }

  pi.on("session_start", async (_event, ctx) => {
    loadedDeferredTools.clear();
    lastDeferredReminderFingerprint = undefined;
    removeToolsFromActiveSet(registeredDeferredTools.keys());
    const activeAtStart = getActiveToolsIfReady();
    if (activeAtStart && earlyConfig.settings?.disableToolSearch !== true && !activeAtStart.includes(TOOL_SEARCH_NAME)) {
      pi.setActiveTools([...activeAtStart, TOOL_SEARCH_NAME]);
    }
    const generation = ++lifecycleGeneration;
    const previousState = state;
    const previousOwner = currentOwner;
    const previousOAuthRuntime = currentOAuthRuntime;
    const owner = createMcpRuntimeOwner();
    const oauthRuntime = createOAuthRuntime(owner.signal);
    currentOwner = owner;
    currentOAuthRuntime = oauthRuntime;
    state = null;
    initPromise = null;

    // Abort synchronously before awaiting cleanup so old callbacks and startup
    // work cannot resume into a stale ExtensionContext.
    const stopPrevious = previousOwner?.stop("MCP extension session restarted") ?? Promise.resolve();
    try {
      await Promise.all([
        stopPrevious,
        shutdownState(previousState, "session_restart"),
        previousOAuthRuntime ? shutdownOAuth(previousOAuthRuntime) : Promise.resolve(),
      ]);
    } catch (error) {
      console.error(`MCP: failed to shut down previous session state: ${formatTerminalError(error)}`);
    }

    if (generation !== lifecycleGeneration || !owner.isActive()) return;

    const initialization = startInitialization(ctx, owner, oauthRuntime, generation, "stale_session_start");
    if (envRaw !== undefined && envRaw !== "__none__") {
      const missingEnvDirectTools = getMissingConfiguredDirectToolServers(
        earlyConfig,
        loadMetadataCache(),
        envDirectToolOverride,
      );
      if (missingEnvDirectTools.length > 0) {
        await initialization;
      }
    }
  });

  pi.on("session_shutdown", async () => {
    loadedDeferredTools.clear();
    lastDeferredReminderFingerprint = undefined;
    ++lifecycleGeneration;
    const currentState = state;
    const owner = currentOwner;
    const oauthRuntime = currentOAuthRuntime;
    currentOwner = null;
    currentOAuthRuntime = null;
    state = null;
    initPromise = null;

    // Abort before awaiting cleanup so delayed initialization cannot touch stale
    // Pi context after session shutdown.
    const stopOwner = owner?.stop("MCP extension session shutdown") ?? Promise.resolve();
    try {
      await Promise.all([
        stopOwner,
        shutdownState(currentState, "session_shutdown"),
        oauthRuntime ? shutdownOAuth(oauthRuntime) : Promise.resolve(),
      ]);
    } catch (error) {
      console.error(`MCP: session shutdown cleanup failed: ${formatTerminalError(error)}`);
    }
  });

  // Re-flag returned MCP tool failures so pi registers them as errors (see toolErrorOverride).
  pi.on("tool_result", (event) => toolErrorOverride(event.details));

  pi.on("before_agent_start", () => {
    if (earlyConfig.settings?.disableToolSearch === true) return;
    const names = [...deferredToolCatalog.keys()]
      .filter((name) => !loadedDeferredTools.has(name))
      .sort((a, b) => a.localeCompare(b));
    if (names.length === 0) return;
    const fingerprint = names.join("\0");
    if (fingerprint === lastDeferredReminderFingerprint) return;
    lastDeferredReminderFingerprint = fingerprint;
    return {
      message: {
        customType: "pi-mcp-adapter:deferred-tools",
        content: buildDeferredToolsReminder(names),
        display: false,
      },
    };
  });

  pi.registerCommand("mcp", {
    description: "Show MCP server status",
    getArgumentCompletions: (prefix: string) => {
      const normalized = prefix.trimStart();
      const argumentMatch = normalized.match(/^(\S+)\s+(.*)$/);
      if (!argumentMatch) {
        const subcommands = [
          { value: "reconnect", label: "reconnect — Reconnect servers" },
          { value: "tools", label: "tools — List all tools" },
          { value: "prompts", label: "prompts — List all MCP prompts" },
          { value: "setup", label: "setup — Configure MCP servers" },
          { value: "logout", label: "logout — Clear server credentials" },
          { value: "disable", label: "disable — Disable a server" },
          { value: "enable", label: "enable — Enable a server" },
          { value: "status", label: "status — Show server status" },
        ].filter(({ value }) => value.startsWith(normalized));
        return subcommands.length > 0 ? subcommands : null;
      }

      const [, subcommand, argumentPrefix] = argumentMatch;
      if (
        (subcommand !== "reconnect" && subcommand !== "logout" && subcommand !== "disable" && subcommand !== "enable")
        || argumentPrefix === undefined
        || !state
      ) return null;

      const servers = Object.keys(state.config.mcpServers)
        .filter((serverName) => serverName.startsWith(argumentPrefix.trimStart()))
        .map((serverName) => ({ value: `${subcommand} ${serverName}`, label: serverName }));
      return servers.length > 0 ? servers : null;
    },
    handler: async (args, ctx) => {
      const commandOwner = currentOwner;
      const commandReload = typeof ctx.reload === "function" ? ctx.reload.bind(ctx) : async () => {};
      const commandHasUI = ctx.hasUI;
      const commandCtx = {
        hasUI: commandHasUI,
        ui: commandHasUI
          ? commandOwner ? createOwnedUi(ctx.ui, commandOwner) : ctx.ui
          : undefined,
        cwd: ctx.cwd,
        mode: ctx.mode,
        signal: commandOwner?.signal ?? ctx.signal,
      } as unknown as ExtensionContext;
      if (!state && initPromise) {
        try {
          const initialized = await initPromise;
          commandOwner?.throwIfInactive();
          state = initialized;
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          if (commandCtx.hasUI) commandCtx.ui?.notify(`MCP initialization failed: ${message}`, "error");
          return;
        }
      }
      if (!state) {
        if (commandCtx.hasUI) commandCtx.ui?.notify("MCP not initialized", "error");
        return;
      }

      const parts = args?.trim()?.split(/\s+/) ?? [];
      const subcommand = parts[0] ?? "";
      const targetServer = parts[1];
      const rest = parts.slice(1).join(" ");

      switch (subcommand) {
        case "reconnect":
          commandOwner?.throwIfInactive();
          await reconnectServers(state, commandCtx, targetServer);
          if (directToolsFrozen) syncToolSurface(commandCtx);
          break;
        case "tools":
          await showTools(state, commandCtx);
          break;
        case "prompts":
          await showPrompts(state, commandCtx);
          break;
        case "setup": {
          commandOwner?.throwIfInactive();
          if (programmaticConfig) {
            commandCtx.ui?.notify("MCP setup is unavailable when config is supplied by createMcpAdapter().", "info");
            break;
          }
          const result = await openMcpSetup(state, pi, commandCtx, earlyConfigPath, "setup");
          if (result?.configChanged) {
            commandOwner?.throwIfInactive();
            await commandReload();
            return;
          }
          break;
        }
        case "logout": {
          const serverName = rest;
          if (!serverName) {
            if (commandCtx.hasUI) commandCtx.ui?.notify("Usage: /mcp logout <server>", "error");
            return;
          }
          commandOwner?.throwIfInactive();
          await logoutServer(serverName, state, commandCtx);
          break;
        }
        case "disable":
        case "enable": {
          const serverName = rest;
          if (programmaticConfig) {
            commandCtx.ui?.notify(`/mcp ${subcommand} is unavailable when config is supplied by createMcpAdapter().`, "info");
            break;
          }
          if (!serverName) {
            commandCtx.ui?.notify(`Usage: /mcp ${subcommand} <server>`, "error");
            break;
          }
          if (!state.config.mcpServers[serverName]) {
            commandCtx.ui?.notify(`Server "${serverName}" not found in effective config`, "error");
            break;
          }
          commandOwner?.throwIfInactive();
          const result = writeProjectServerDisabledOverride(earlyConfigPath, commandCtx.cwd, serverName, subcommand === "disable");
          if (result.changed) {
            commandCtx.ui?.notify(`${subcommand === "disable" ? "Disabled" : "Enabled"} server "${serverName}" in ${result.path} — run /reload to apply`, "info");
          } else {
            commandCtx.ui?.notify(`Server "${serverName}" is already ${subcommand === "disable" ? "disabled" : "enabled"}`, "info");
          }
          break;
        }
        case "status":
        case "":
        default:
          if (commandCtx.hasUI) {
            commandOwner?.throwIfInactive();
            if (programmaticConfig) {
              commandCtx.ui?.notify("MCP status is shown from the in-memory SDK config; configuration discovery is unavailable.", "info");
              await showStatus(state, commandCtx);
              break;
            }
            const result = await openMcpPanel(state, pi, commandCtx, earlyConfigPath, (changes) => {
              applyDirectToolConfigChanges(changes);
              syncToolSurface(commandCtx);
            });
            if (result?.configChanged) {
              commandOwner?.throwIfInactive();
              await commandReload();
              return;
            }
          } else {
            await showStatus(state, commandCtx);
          }
          break;
      }
    },
  });

  pi.registerCommand("mcp-auth", {
    description: "Authenticate with an MCP server (OAuth)",
    handler: async (args, ctx) => {
      const commandOwner = currentOwner;
      const commandHasUI = ctx.hasUI;
      const commandCtx = {
        hasUI: commandHasUI,
        ui: commandHasUI
          ? commandOwner ? createOwnedUi(ctx.ui, commandOwner) : ctx.ui
          : undefined,
        cwd: ctx.cwd,
        mode: ctx.mode,
        signal: commandOwner?.signal ?? ctx.signal,
      } as unknown as ExtensionContext;
      const serverName = args?.trim();
      if (!serverName && !commandCtx.hasUI) {
        return;
      }

      if (!state && initPromise) {
        try {
          const initialized = await initPromise;
          commandOwner?.throwIfInactive();
          state = initialized;
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          if (commandCtx.hasUI) commandCtx.ui?.notify(`MCP initialization failed: ${message}`, "error");
          return;
        }
      }
      if (!state) {
        if (commandCtx.hasUI) commandCtx.ui?.notify("MCP not initialized", "error");
        return;
      }

      if (!serverName) {
        if (programmaticConfig) {
          commandCtx.ui?.notify("Use /mcp-auth <server> to authenticate a server from the in-memory SDK config.", "info");
          return;
        }
        await openMcpAuthPanel(state, pi, commandCtx, earlyConfigPath);
        return;
      }

      const result = await authenticateServer(serverName, state.config, commandCtx, commandCtx.signal, state.oauthRuntime);
      if (result.ok) {
        commandOwner?.throwIfInactive();
        await reconnectServer(state, commandCtx, serverName);
      }
    },
  });

  if (earlyConfig.settings?.scriptMode !== false) {
    (pi.registerTool as (tool: unknown) => unknown)({
      name: "mcpScript",
      label: "MCP Script",
      description: "Run trusted JavaScript that makes multiple MCP tool calls in one request — loop, filter, chain, or fan out between calls. For a single MCP call, search, describe, status check, or auth action, use the mcp tool instead. Discover with await tools.search({ query }) — resolves to { items: [{ path, name, server, description? }], total, hasMore, nextOffset }, not an { ok, data } envelope. Inspect with await tools.describe({ path }) — resolves to the tool descriptor with inputTypeScript, or { path, error: { code, message, suggestions } }. Then call tools.call(path, args) — resolves to { ok: true, data } or { ok: false, error: { code, message } } — or use direct flat calls when the name is already known; use emit(value) for user-visible output. Load the mcp-scripting skill for the full workflow guide.",
      promptSnippet: "Batch multiple MCP tool calls in one JavaScript request (loop, filter, chain)",
      parameters: Type.Object({
        code: Type.String({ description: "Trusted JavaScript MCP script. Use tools.<prefixedToolName>(args) and emit(value)." }),
        timeoutMs: optionalNumber({ minimum: 1, description: "Execution timeout in milliseconds (default: 30000)" }),
      }),
      renderResult: renderMcpToolResult,
      async execute(_toolCallId: string, params: { code: string; timeoutMs?: number }, signal: AbortSignal | undefined) {
        const executeOwner = currentOwner;
        if (!state && initPromise) {
          try {
            const initialized = await awaitWithTimeout(initPromise, INIT_WAIT_TIMEOUT_MS);
            if (initialized === INIT_WAIT_TIMED_OUT) {
              return {
                content: [{ type: "text" as const, text: "MCP initialization is still in progress. Try again shortly." }],
                details: { mode: "script", error: "init_timeout", timeoutMs: INIT_WAIT_TIMEOUT_MS },
              };
            }
            executeOwner?.throwIfInactive();
            state = initialized;
          } catch (error) {
            if (executeOwner && isAbortError(error, executeOwner.signal)) throw error;
            const message = error instanceof Error ? error.message : String(error);
            return {
              content: [{ type: "text" as const, text: `MCP initialization failed: ${message}` }],
              details: { mode: "script", error: "init_failed", message },
            };
          }
        }
        if (!state) {
          return {
            content: [{ type: "text" as const, text: "MCP not initialized" }],
            details: { mode: "script", error: "not_initialized" },
          };
        }
        executeOwner?.throwIfInactive();
        return runMcpScript(state, params.code, params.timeoutMs, getPiTools, signal);
      },
    });
  }

  if (earlyConfig.settings?.disableToolSearch !== true) {
    (pi.registerTool as (tool: unknown) => unknown)({
      name: TOOL_SEARCH_NAME,
      label: "Tool Search",
      description: TOOL_SEARCH_DESCRIPTION,
      promptSnippet: "Load deferred MCP tools by exact name or capability search",
      parameters: Type.Object({
        query: Type.String({ description: "Search query, +required-name query, or select:ToolA,ToolB" }),
        max_results: requiredNumber({
          minimum: 1,
          maximum: 100,
          default: DEFAULT_TOOL_SEARCH_RESULTS,
          description: "Maximum number of matching tools to load (default: 5)",
        }),
      }),
      async execute(_toolCallId: string, params: { query: string; max_results: number }, signal: AbortSignal | undefined) {
        const executeOwner = currentOwner;
        if (!state && initPromise) {
          try {
            const initialized = await awaitWithTimeout(initPromise, INIT_WAIT_TIMEOUT_MS);
            if (initialized === INIT_WAIT_TIMED_OUT) {
              return {
                content: [{ type: "text" as const, text: "MCP initialization is still in progress. Try again shortly." }],
                details: { query: params.query, matches: [], error: "init_timeout" },
              };
            }
            executeOwner?.throwIfInactive();
            state = initialized;
          } catch (error) {
            if (executeOwner && isAbortError(error, executeOwner.signal)) throw error;
            const message = error instanceof Error ? error.message : String(error);
            return {
              content: [{ type: "text" as const, text: `MCP initialization failed: ${message}` }],
              details: { query: params.query, matches: [], error: "init_failed", message },
            };
          }
        }
        if (signal?.aborted) throw signal.reason;

        const matches = searchDeferredTools(
          [...deferredToolCatalog.values()],
          params.query,
          params.max_results ?? DEFAULT_TOOL_SEARCH_RESULTS,
        );
        const active = pi.getActiveTools();
        const activeSet = new Set(active);
        const added = matches.map((match) => match.name).filter((name) => !activeSet.has(name));
        for (const match of matches) loadedDeferredTools.add(match.name);
        if (added.length > 0) {
          // Pi records a purely additive set change as addedToolNames. Its
          // Anthropic adapter then emits typed tool_reference blocks and sends
          // only these definitions with defer_loading:true, matching Claude Code.
          pi.setActiveTools([...active, ...added]);
        }

        const names = matches.map((match) => match.name);
        return {
          content: [{
            type: "text" as const,
            text: names.length > 0
              ? `Loaded tools: ${names.join(", ")}`
              : `No deferred tools found for: ${params.query}`,
          }],
          details: {
            query: params.query,
            matches: names,
            total_mcp_tools: deferredToolCatalog.size,
          },
        };
      },
    });
  }

  function registerProxyTool(description: string): void {
    (pi.registerTool as (tool: unknown) => unknown)({
      name: "mcp",
      label: "MCP",
      description,
      promptSnippet: "MCP gateway — status, search, describe, auth, and single MCP tool calls",
      renderShell: toolRenderShell,
      renderCall: createMcpProxyToolCallRenderer(toolRenderOptions),
      parameters: Type.Object({
        tool: Type.Optional(Type.String({ description: "Tool name to call (e.g., 'xcodebuild_list_sims')" })),
        args: Type.Optional(Type.Union([
          Type.String({ description: "Arguments as a JSON string (e.g., '{\"key\": \"value\"}')" }),
          Type.Object({}, {
            additionalProperties: true,
            description: 'Arguments as a JSON object (e.g., { "key": "value" })',
          }),
        ], { description: "Tool arguments as a JSON object, or as a JSON string encoding one" })),
        connect: Type.Optional(Type.String({ description: "Server name to connect (lazy connect + metadata refresh)" })),
        describe: Type.Optional(Type.String({ description: "Tool name to describe (shows parameters)" })),
        instructions: Type.Optional(Type.String({ description: "Server name to show that server's usage instructions" })),
        search: Type.Optional(Type.String({ description: "Search tools by name/description" })),
        regex: Type.Optional(Type.Boolean({ description: "Treat search as regex (default: substring match)" })),
        includeSchemas: Type.Optional(Type.Boolean({ description: "Include parameter schemas in search results (default: true)" })),
        limit: optionalNumber({ minimum: 1, description: "Maximum search results to return (default: 12)" }),
        offset: optionalNumber({ minimum: 0, description: "Search result offset (default: 0)" }),
        server: Type.Optional(Type.String({ description: "Filter to specific server (also disambiguates tool calls)" })),
        action: Type.Optional(Type.String({ description: "Action: 'ui-messages', 'auth-start', or 'auth-complete'" })),
      }),
      renderResult: renderMcpToolResult,
      async execute(_toolCallId: string, params: {
        tool?: string;
        args?: string | Record<string, unknown>;
        connect?: string;
        describe?: string;
        instructions?: string;
        search?: string;
        regex?: boolean;
        includeSchemas?: boolean;
        limit?: number;
        offset?: number;
        server?: string;
        action?: string;
      }, signal: AbortSignal | undefined, _onUpdate: AgentToolUpdateCallback<Record<string, unknown>> | undefined, _ctx: ExtensionContext) {
        const executeOwner = currentOwner;
        const parseArgs = (value: string | Record<string, unknown> | undefined): Record<string, unknown> | undefined => {
          if (value === undefined || value === "") return undefined;
          let args: unknown;
          if (typeof value === "string") {
            try {
              args = JSON.parse(value);
            } catch (error) {
              if (error instanceof SyntaxError) {
                throw new Error(`Invalid args JSON: ${error.message}`, { cause: error });
              }
              throw error;
            }
          } else {
            args = value;
          }

          if (typeof args !== "object" || args === null || Array.isArray(args)) {
            const gotType = Array.isArray(args) ? "array" : args === null ? "null" : typeof args;
            throw new Error(`Invalid args: expected a JSON object, got ${gotType}`);
          }
          return args as Record<string, unknown>;
        };
        let parsedArgs = parseArgs(params.args);
        let dispatchParams = params;
        const hasGatewayMode = (value: typeof params): boolean =>
          value.tool !== undefined
          || value.connect !== undefined
          || value.describe !== undefined
          || value.instructions !== undefined
          || value.search !== undefined
          || value.server !== undefined
          || value.action !== undefined;
        if (!hasGatewayMode(params) && parsedArgs) {
          const nestedParams = parsedArgs as typeof params;
          if (hasGatewayMode(nestedParams)) {
            dispatchParams = nestedParams;
            parsedArgs = parseArgs(nestedParams.args);
          } else {
            throw new Error("Gateway params were nested inside `args`; pass them top-level (for example, mcp({ search: \"...\" }) or mcp({ tool: \"...\", args: {} })).");
          }
        } else if (!hasGatewayMode(params) && params.args !== undefined) {
          throw new Error("Gateway params were nested inside `args`; pass them top-level (for example, mcp({ search: \"...\" }) or mcp({ tool: \"...\", args: {} })).");
        }

        if (!state && initPromise) {
          try {
            const initialized = await awaitWithTimeout(initPromise, INIT_WAIT_TIMEOUT_MS);
            if (initialized === INIT_WAIT_TIMED_OUT) {
              return {
                content: [{ type: "text" as const, text: "MCP initialization is still in progress. Try again shortly." }],
                details: { error: "init_timeout", timeoutMs: INIT_WAIT_TIMEOUT_MS },
              };
            }
            executeOwner?.throwIfInactive();
            state = initialized;
          } catch (error) {
            if (executeOwner && isAbortError(error, executeOwner.signal)) throw error;
            const message = error instanceof Error ? error.message : String(error);
            return {
              content: [{ type: "text" as const, text: `MCP initialization failed: ${message}` }],
              details: { error: "init_failed", message },
            };
          }
        }
        if (!state) {
          return {
            content: [{ type: "text" as const, text: "MCP not initialized" }],
            details: { error: "not_initialized" },
          };
        }
        executeOwner?.throwIfInactive();

        if (dispatchParams.action === "ui-messages") {
          return executeUiMessages(state);
        }
        if (dispatchParams.action === "auth-start") {
          if (!dispatchParams.server) {
            return {
              content: [{ type: "text" as const, text: "auth-start requires `server`. Example: mcp({ action: \"auth-start\", server: \"linear-server\" })" }],
              details: { mode: "auth-start", error: "missing_server" },
            };
          }
          return signal
            ? executeAuthStart(state, dispatchParams.server, signal)
            : executeAuthStart(state, dispatchParams.server);
        }
        if (dispatchParams.action === "auth-complete") {
          if (!dispatchParams.server) {
            return {
              content: [{ type: "text" as const, text: "auth-complete requires `server`." }],
              details: { mode: "auth-complete", error: "missing_server" },
            };
          }
          const input = parsedArgs?.redirectUrl ?? parsedArgs?.code ?? parsedArgs?.input;
          if (typeof input !== "string" || input.trim().length === 0) {
            return {
              content: [{ type: "text" as const, text: "auth-complete requires args with `redirectUrl`, `code`, or `input`." }],
              details: { mode: "auth-complete", error: "missing_input" },
            };
          }
          return signal
            ? executeAuthComplete(state, dispatchParams.server, input, signal)
            : executeAuthComplete(state, dispatchParams.server, input);
        }
        if (dispatchParams.tool) {
          return executeCall(state, dispatchParams.tool, parsedArgs, dispatchParams.server, getPiTools, signal);
        }
        if (dispatchParams.connect) {
          const result = await executeConnect(state, dispatchParams.connect, signal);
          syncToolSurface(_ctx as ExtensionContext);
          return result;
        }
        if (dispatchParams.describe) {
          return executeDescribe(state, dispatchParams.describe);
        }
        if (dispatchParams.instructions) {
          return executeInstructions(state, dispatchParams.instructions);
        }
        if (dispatchParams.search !== undefined) {
          return executeSearch(state, dispatchParams.search, dispatchParams.regex, dispatchParams.server, dispatchParams.includeSchemas, dispatchParams.limit, dispatchParams.offset);
        }
        if (dispatchParams.server) {
          return executeList(state, dispatchParams.server);
        }
        return executeStatus(state);
      },
    });
    proxyToolRegistered = true;
    proxyToolDescription = description;
  }

  function syncProxyTool(config: McpConfig, cache: MetadataCache | null, directSpecs: DirectToolSpec[]): void {
    const missingConfiguredDirectToolServers = getMissingConfiguredDirectToolServers(
      config,
      cache,
      envRaw === undefined || envRaw === "__none__" ? undefined : envDirectToolOverride,
    );
    const shouldRegisterProxyTool =
      config.settings?.disableProxyTool !== true
      || directSpecs.length === 0
      || missingConfiguredDirectToolServers.length > 0;

    if (shouldRegisterProxyTool) {
      const description = buildProxyDescription(config, cache, directSpecs);
      if (!proxyToolRegistered || proxyToolDescription !== description) {
        registerProxyTool(description);
        return;
      }
      const activeTools = getActiveToolsIfReady();
      if (activeTools && !activeTools.includes("mcp")) {
        pi.setActiveTools([...activeTools, "mcp"]);
      }
      return;
    }

    if (proxyToolRegistered) {
      const unregistered = deactivateTools(["mcp"]);
      if (unregistered.includes("mcp")) {
        proxyToolRegistered = false;
        proxyToolDescription = null;
      }
    }
  }

  const initialDirectTools = syncDirectTools(earlyConfig, earlyCache).specs;
  syncDeferredTools(earlyConfig, earlyCache, initialDirectTools);
  syncProxyTool(earlyConfig, earlyCache, initialDirectTools);
  startLoadTimeInitialization();
}

export function createMcpAdapter(options: McpAdapterOptions = {}) {
  const factoryConfig = options.config !== undefined ? cloneMcpConfig(options.config) : undefined;
  return function mcpAdapter(pi: ExtensionAPI) {
    installMcpAdapter(pi, {
      ...(options.configPath !== undefined ? { configPath: options.configPath } : {}),
      ...(factoryConfig !== undefined ? { config: cloneMcpConfig(factoryConfig) } : {}),
    });
  };
}

export default createMcpAdapter();
