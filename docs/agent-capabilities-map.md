# Conductor-Tasks: Agent Capabilities Map & Multi-LLM Enhancement Analysis

**Branch:** `claude/map-agent-capabilities-30bYz`
**Date:** 2026-03-13
**Scope:** Deep inspection of the LLM abstraction layer, MCP tool surface, routing system, and gaps for diverse Agent Team capabilities.

---

## 1. System Architecture Overview

Conductor-Tasks is a **task management MCP server + CLI** designed for AI-driven development workflows. It exposes 22 MCP tools that AI agents can call, and internally delegates AI work to a configurable pool of LLM providers. It runs in two modes:

- **MCP Server mode** (`--serve-mcp`): talks to IDEs (Cursor, Windsurf, Roo-Code, Cline) over stdio using the Model Context Protocol.
- **CLI mode**: direct command execution via `yargs`.

The three runtime singletons are:

```
LLMManager      ← orchestrates all LLM providers (routing, retries, rate limits)
ContextManager  ← manages project context items, anchor points, IDE rules
TaskManager     ← owns TASKS.md, exposes task CRUD + AI-powered generation
```

---

## 2. LLM Provider Inventory

### 2.1 Registered Providers

| Provider     | SDK / Transport             | Default Model                         | Env Key(s)                                  | Interface   |
|--------------|-----------------------------|---------------------------------------|---------------------------------------------|-------------|
| `anthropic`  | `@anthropic-ai/sdk`          | `claude-3.7-sonnet-20240607`          | `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY`      | `LLMClient` |
| `openai`     | `openai`                    | `gpt-4o`                              | `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`     | `LLMClient` |
| `groq`       | `groq-sdk`                  | `deepseek-r1-distill-llama-70b`       | `GROQ_API_KEY`                              | `LLMClient` |
| `mistral`    | `@mistralai/mistralai`      | `mistral-large-latest`                | `MISTRAL_API_KEY`                           | `LLMClient` |
| `mixtral`    | `@mistralai/mistralai`      | `mixtral-8x7b-32768`                  | `MIXTRAL_API_KEY`                           | `LLMProvider` (legacy) |
| `gemini`     | `@google/generative-ai`     | `gemini-2.5-pro-exp-03-25`            | `GEMINI_API_KEY`                            | `LLMClient` |
| `xai`        | `openai` (compatible)       | `grok-3`                              | `XAI_API_KEY`                               | `LLMClient` |
| `ollama`     | `node-fetch` (HTTP)         | `llama3`                              | `OLLAMA_ENABLED=true` / `OLLAMA_API_KEY`    | `LLMClient` + `LLMProvider` |
| `perplexity` | `openai` (compatible)       | `llama-3-sonar-medium-32k-online`     | `PERPLEXITY_API_KEY`                        | `LLMClient` + `LLMProvider` |
| `openrouter` | `openai` (compatible)       | `mistralai/mistral-7b-instruct`       | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`    | `LLMClient` |

**Fallback priority order** (when no provider is specified):
```
anthropic → gemini → openai → groq → mistral → mixtral → ollama → perplexity → openrouter → xai
```

### 2.2 Two Competing Provider Interfaces

There are **two co-existing interface patterns**, which creates inconsistency:

**`LLMClient`** (primary, `src/llm/types.ts`):
```typescript
interface LLMClient {
  complete(options: LLMCompletionOptions): Promise<LLMCompletionResult>;
  getProviderName(): string;
  getModelName(): string;
}
```

**`LLMProvider`** (legacy, `src/core/types.ts`):
```typescript
interface LLMProvider {
  name: string;
  generate(request: LLMRequest): Promise<LLMResponse>;
  isAvailable(): boolean;
}
```

`LLMManager.executeRequestInternal()` handles both via duck-typing:
```typescript
if (providerObj.generate) {
  result = await providerObj.generate(request);        // LLMProvider path
} else if (providerObj.client) {
  result = await client.complete({...});               // LLMClient path
}
```

`Ollama` and `Perplexity` each have **dual implementations** (both `*Provider` and `*Client` classes), but only the `*Client` is registered in `LLMManager`. The `*Provider` classes are unused dead code.

### 2.3 The `clientFactory.ts` Gap

`src/llm/clientFactory.ts` is a simplified factory used by some command handlers. It only knows about **Anthropic and OpenAI**, despite 10 providers existing in `LLMManager`. This is a significant inconsistency — any code using `getLLMClient()` instead of `llmManager.sendRequest()` is silently limited to two providers.

---

## 3. LLM Manager Deep Dive

### 3.1 Completion Options Exposed

```typescript
interface LLMCompletionOptions {
  prompt: string;
  systemPrompt?: string;
  maxTokens?: number;         // default: 4000
  temperature?: number;       // default: 0.7
  topP?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  stopSequences?: string[];
  stream?: boolean;
  onStreamUpdate?: (chunk: string) => void;
}
```

All providers support `stream` + `onStreamUpdate` in their `*Client` implementations. However, **no MCP tool handler currently uses streaming** — all calls go through `llmManager.sendRequest()` which does not expose stream callbacks.

### 3.2 Concurrency & Rate-Limit Controls

| Setting                          | Default | Env Override                        |
|----------------------------------|---------|--------------------------------------|
| Max concurrent requests          | 5       | `LLM_MAX_CONCURRENT_REQUESTS`       |
| Max retries per provider         | 3       | `LLM_MAX_RETRIES`                   |
| Max provider attempts (fallback) | 3       | `LLM_MAX_PROVIDER_ATTEMPTS`         |
| Base rate-limit backoff          | 60s     | `LLM_BASE_RATE_LIMIT_DURATION_MS`   |
| Max rate-limit backoff           | 5 min   | `LLM_MAX_RATE_LIMIT_DURATION_MS`    |

Requests are queued internally. Rate-limited providers are marked with exponential backoff and skipped in subsequent routing.

### 3.3 Task-to-Provider Routing

The `taskToProviderMap` allows env-var driven assignment of named tasks to specific providers:

```bash
# Route parse-prd to Anthropic, research to Perplexity
ANTHROPIC_TASKS="initialize-project, parse-prd, expand-task"
PERPLEXITY_TASKS="research-topic"
GROQ_TASKS="suggest-task-improvements"
```

Routing logic in `getProviderForTask()`:
1. Exact match on task name
2. Normalized match (strips `-` and `_`)
3. Partial/substring match (bidirectional)
4. Fall back to default provider

`sendRequest()` accepts an optional `taskName` field on the `LLMRequest` to trigger this routing.

### 3.4 Global Config Overrides

These env vars apply globally across all providers:
```
TEMPERATURE, MAX_TOKENS, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY
DEFAULT_LLM_PROVIDER
```

Per-request overrides can be passed in `request.options`.

### 3.5 Prompt Refinement Service

`src/core/promptRefinementService.ts` provides a **meta-LLM service**: given a failed prompt + failed response + desired spec, it generates an improved prompt. Uses `llmManager.sendRequest()` with `temperature: 0.05`. This is not currently wired to any auto-retry loop — it's a standalone utility.

---

## 4. MCP Tool Surface (22 Tools)

### 4.1 Pure Task Management (No LLM)

| Tool | Description |
|------|-------------|
| `create-task` | Create task with title, description, priority, status, assignee, tags, complexity, dependencies |
| `update-task` | Patch any task field |
| `list-tasks` | Filter by status/priority/tags, sort |
| `get-task` | Fetch single task by ID |
| `delete-task` | Remove task |
| `add-task-note` | Append progress/comment/blocker/solution note |
| `get-next-task` | Priority-based next task selector |
| `visualize-tasks-kanban` | Kanban board text rendering |
| `visualize-tasks-dependency-tree` | Dependency tree text rendering |
| `visualize-tasks-dashboard` | Summary stats dashboard |
| `list-task-templates` | List available templates |
| `get-task-template` | Fetch template by name |
| `create-task-from-template` | Instantiate template with variable substitution |
| `propose-diff` | Acknowledge a proposed diff (does not apply it) |

### 4.2 AI-Powered Tools (Use LLMManager)

| Tool | LLM Use | Provider Routing Key |
|------|---------|---------------------|
| `parse-prd` | Parse PRD text → structured tasks JSON | `"parse-prd"` |
| `parse-prd-file` | Read file + parse PRD | `"parse-prd-file"` |
| `initialize-project` | Generate project context + TASKS.md scaffold | `"initialize-project"` |
| `generate-implementation-steps` | Generate ordered steps for a task | `"generate-implementation-steps"` |
| `expand-task` | Generate subtasks + detailed breakdown | `"expand-task"` |
| `suggest-task-improvements` | AI critique + improvement suggestions | `"suggest-task-improvements"` |
| `help-implement-task` | Pair-programmer guidance with code snippets | `"help-implement-task"` |
| `research-topic` | Research query; prefers Perplexity, falls back to others | `"research-topic"` |
| `generate-diff` | Generate a diff patch for a file | `"generate-diff"` |

---

## 5. Identified Gaps & Limitations

### 5.1 Architectural Gaps

| Gap | Location | Impact |
|-----|----------|--------|
| `clientFactory.ts` only covers Anthropic + OpenAI | `src/llm/clientFactory.ts` | Code using this factory ignores 8 other providers |
| Dual interfaces (`LLMClient` vs `LLMProvider`) | Multiple files | Inconsistent provider wrapping, dead code for Ollama/Perplexity Provider classes |
| No tool-use / function-calling abstraction | `LLMCompletionOptions` | Cannot leverage structured tool calls from OpenAI, Gemini, Anthropic |
| No multi-turn conversation support | All handlers | All LLM calls are single-shot; no conversation history threaded through |
| No agent role/persona assignment per provider | `LLMManager` | Cannot configure "Provider X acts as Reviewer, Provider Y as Architect" |
| No capability metadata per provider | `LLMManager` | No way to query "which providers support streaming / function-calling / web-search" |
| Stream not exposed to MCP tools | All handlers | Streaming responses can't be forwarded to IDE clients |
| Task-to-provider routing is env-var only | `LLMManager.loadTaskProviderMappings()` | No programmatic or dynamic routing API |

### 5.2 Missing Providers

The following providers have no implementation yet:

- **Cohere** (Command R+) — strong at RAG/retrieval tasks
- **Together.ai** — wide open-source model catalog
- **DeepSeek** — high reasoning capability, cost-effective
- **Cerebras** — ultra-fast inference for latency-sensitive tasks
- **Fireworks AI** — fast open-source inference
- **Bedrock (AWS)** — enterprise/VPC deployments
- **Azure OpenAI** — enterprise OpenAI with compliance controls
- **Vertex AI (Google)** — enterprise Gemini with fine-tuning
- **Coze / custom agent platforms** — workflow-level agents

### 5.3 Agent Team Capability Gaps

The system has no concept of **agent roles**, **agent teams**, or **agent coordination**. Specifically missing:

1. **Role assignment**: No way to declare "Anthropic = Architect, Groq = Fast Reviewer, Perplexity = Researcher"
2. **Sequential agent chaining**: No pipeline where one agent's output feeds the next
3. **Parallel agent dispatch**: No fan-out to multiple providers for consensus/voting
4. **Agent memory**: No persistent per-agent conversation history
5. **Agent specialization metadata**: No capability flags (e.g., `supportsToolCalls`, `hasWebSearch`, `contextWindow`)
6. **Structured output contracts**: No schema enforcement for inter-agent data exchange (beyond ad-hoc JSON prompting)
7. **Agent feedback loops**: `refinePrompt` exists but is not wired into any auto-correction cycle
8. **Task-type → capability matching**: No logic to route `research-topic` to web-search-capable providers automatically (it's hardcoded to try Perplexity first)

---

## 6. Enhancement Roadmap: Diverse Agent Team Capabilities

### Phase 1 — Unify the Provider Interface

**Goal**: Eliminate the dual-interface inconsistency and bring all providers to parity.

```typescript
// Proposed unified interface
interface AgentProvider {
  readonly name: string;
  readonly capabilities: ProviderCapabilities;
  complete(options: AgentCompletionOptions): Promise<AgentCompletionResult>;
  isAvailable(): boolean;
  getDefaultModel(): string;
}

interface ProviderCapabilities {
  supportsStreaming: boolean;
  supportsToolCalls: boolean;       // function/tool calling
  supportsVision: boolean;          // multimodal
  hasWebSearch: boolean;            // live web access (Perplexity, Gemini with grounding)
  maxContextWindow: number;         // tokens
  supportedModalities: ('text' | 'image' | 'audio')[];
}
```

Fix `clientFactory.ts` to delegate to `LLMManager` or be removed entirely.

### Phase 2 — Agent Role & Team Configuration

**Goal**: Allow named agent roles with provider + model + persona assignments.

```typescript
interface AgentRole {
  name: string;                     // e.g., "architect", "reviewer", "researcher"
  provider: string;                 // e.g., "anthropic"
  model?: string;                   // override default model
  systemPrompt?: string;            // role-specific persona/instructions
  temperature?: number;             // role-specific temperature
  capabilities?: string[];          // required capability flags
}

interface AgentTeam {
  roles: AgentRole[];
  defaultRole: string;
  taskRoutingRules: TaskRoutingRule[];  // maps task types to roles
}
```

Configuration via `CONDUCTOR_AGENT_TEAM` env var (JSON) or a `conductor.config.json` file.

Example team config:
```json
{
  "roles": [
    { "name": "architect",  "provider": "anthropic", "model": "claude-opus-4-6",         "temperature": 0.3 },
    { "name": "researcher", "provider": "perplexity", "model": "llama-3-sonar-large-32k-online" },
    { "name": "fast-coder", "provider": "groq",       "model": "deepseek-r1-distill-llama-70b", "temperature": 0.1 },
    { "name": "reviewer",   "provider": "gemini",     "model": "gemini-2.5-pro-exp-03-25",      "temperature": 0.2 },
    { "name": "local",      "provider": "ollama",     "model": "codellama:34b" }
  ],
  "taskRoutingRules": [
    { "taskNamePattern": "parse-prd*",              "role": "architect" },
    { "taskNamePattern": "research-topic",          "role": "researcher" },
    { "taskNamePattern": "generate-*",              "role": "fast-coder" },
    { "taskNamePattern": "suggest-task-improvements","role": "reviewer" }
  ]
}
```

### Phase 3 — Tool-Call / Function-Calling Abstraction

**Goal**: Expose structured tool calls through the provider abstraction so AI agents can use tools (file read, web search, code execution) natively.

```typescript
interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;  // JSON Schema
}

interface AgentCompletionOptions extends LLMCompletionOptions {
  tools?: ToolDefinition[];
  toolChoice?: 'auto' | 'none' | { name: string };
}

interface AgentCompletionResult extends LLMCompletionResult {
  toolCalls?: ToolCall[];
}
```

Each provider adapter maps this to its native API:
- Anthropic: `tools` + `tool_use` blocks
- OpenAI/Groq/xAI: `tools` + `tool_calls` in response
- Gemini: `tools` with `functionDeclarations`
- Ollama: `/api/chat` with `tools` (requires Ollama ≥ 0.3)

### Phase 4 — Agent Pipeline / Orchestration

**Goal**: Support sequential and parallel agent execution patterns.

**Sequential chain** (output of agent N → input of agent N+1):
```typescript
class AgentPipeline {
  addStep(role: string, promptBuilder: (prevResult: string) => string): this;
  execute(initialInput: string): Promise<PipelineResult>;
}
```

Use case: `researcher → architect → fast-coder → reviewer` for full task implementation.

**Parallel fan-out** (same prompt → multiple agents → aggregate):
```typescript
class AgentConsensus {
  addVoter(role: string): this;
  aggregate(strategy: 'first' | 'longest' | 'majority-vote' | 'synthesis'): this;
  execute(prompt: string): Promise<ConsensusResult>;
}
```

Use case: `suggest-task-improvements` sent to three different providers; synthesize best suggestions.

### Phase 5 — Multi-Turn Conversation Support

**Goal**: Thread conversation history through agent calls for iterative refinement.

```typescript
interface ConversationMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface AgentSession {
  id: string;
  agentRole: string;
  history: ConversationMessage[];
  sendMessage(content: string): Promise<string>;
}
```

Persist sessions in memory (or optionally to disk) keyed by `taskId` so the same agent can be re-engaged across multiple tool calls within a task.

### Phase 6 — New Provider Additions

Priority order for new provider integrations (each follows the existing `LLMClient` pattern):

1. **Azure OpenAI** — enterprise requirement; uses `openai` SDK with `baseURL` override (already partially supported via `OPENAI_API_BASE_URL`)
2. **AWS Bedrock** — requires `@aws-sdk/client-bedrock-runtime`; covers Titan, Llama, Mistral on AWS
3. **Cohere** — Command R+ for RAG; uses `cohere-ai` SDK
4. **DeepSeek** — OpenAI-compatible API; zero new SDK needed, just `baseURL` override via `openai`
5. **Together.ai** — OpenAI-compatible; same pattern as DeepSeek
6. **Fireworks AI** — OpenAI-compatible; same pattern
7. **Cerebras** — custom SDK or REST; for ultra-low latency tasks

---

## 7. Environment Variable Reference (Complete)

### Existing Variables

```bash
# Provider API Keys
ANTHROPIC_API_KEY / CLAUDE_API_KEY
OPENAI_API_KEY
OPENAI_API_BASE_URL / LLM_PROVIDER_OPENAI_BASE_URL
GROQ_API_KEY
MISTRAL_API_KEY
MIXTRAL_API_KEY
GEMINI_API_KEY
XAI_API_KEY
OLLAMA_ENABLED=true / OLLAMA_API_KEY
OLLAMA_BASE_URL           # default: http://localhost:11434
OLLAMA_MODEL              # default: llama3
PERPLEXITY_API_KEY
PERPLEXITY_API_BASE_URL   # default: https://api.perplexity.ai
PERPLEXITY_MODEL
OPENROUTER_API_KEY
OPENROUTER_MODEL

# Per-provider model overrides
ANTHROPIC_MODEL
MODEL                     # used by OpenAI client as fallback

# LLM Manager Tuning
DEFAULT_LLM_PROVIDER
LLM_MAX_RETRIES           # default: 3
LLM_MAX_PROVIDER_ATTEMPTS # default: 3
LLM_MAX_CONCURRENT_REQUESTS # default: 5
LLM_BASE_RATE_LIMIT_DURATION_MS # default: 60000
LLM_MAX_RATE_LIMIT_DURATION_MS  # default: 300000

# Global Completion Defaults
TEMPERATURE
MAX_TOKENS
TOP_P
FREQUENCY_PENALTY
PRESENCE_PENALTY

# Task-to-Provider Routing
ANTHROPIC_TASKS="task1, task2"
OPENAI_TASKS="task3"
GROQ_TASKS="task4"
# (pattern: {PROVIDER_UPPERCASE}_TASKS)

# Task Manager
WORKSPACE_FOLDER_PATHS    # semicolon-separated; first path used
TASKS_FILENAME            # default: TASKS.md
DEFAULT_SUBTASKS          # default: 3
DEFAULT_PRIORITY          # default: medium

# IDE Integration
IDE                       # cursor | windsurf | roo-code | cline | generic
MCP_MODE                  # set to "true" automatically in MCP mode
```

### Proposed New Variables (Phase 2+)

```bash
# Agent Team Config
CONDUCTOR_AGENT_TEAM_CONFIG   # path to JSON config file
CONDUCTOR_AGENT_TEAM_JSON     # inline JSON agent team definition

# Per-role API key overrides (if same provider used for multiple roles)
CONDUCTOR_ROLE_ARCHITECT_PROVIDER=anthropic
CONDUCTOR_ROLE_RESEARCHER_PROVIDER=perplexity
CONDUCTOR_ROLE_FAST_CODER_PROVIDER=groq
CONDUCTOR_ROLE_REVIEWER_PROVIDER=gemini

# New provider keys
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
AWS_BEDROCK_REGION
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
COHERE_API_KEY
DEEPSEEK_API_KEY
TOGETHER_API_KEY
FIREWORKS_API_KEY
CEREBRAS_API_KEY
```

---

## 8. Key Files Reference

| File | Role |
|------|------|
| `src/llm/llmManager.ts` | Central provider registry, routing, rate limiting, concurrency queue |
| `src/llm/types.ts` | `LLMClient`, `LLMCompletionOptions`, `LLMCompletionResult` interfaces |
| `src/core/types.ts` | `LLMProvider` (legacy), `Task`, `TaskStatus`, `ToolResultWithNextSteps`, `SuggestedAction` |
| `src/llm/clientFactory.ts` | Simplified factory (only Anthropic + OpenAI) — **needs update** |
| `src/llm/providers/anthropic.ts` | Anthropic SDK client with streaming + JSON mode |
| `src/llm/providers/openai.ts` | OpenAI SDK client with custom `baseURL` support |
| `src/llm/providers/openrouter.ts` | OpenRouter via OpenAI-compat; any model via single key |
| `src/llm/providers/ollama.ts` | Local Ollama via HTTP; both `LLMClient` and `LLMProvider` (latter unused) |
| `src/llm/providers/perplexity.ts` | Perplexity via OpenAI-compat; both classes (latter unused) |
| `src/core/promptRefinementService.ts` | Meta-LLM prompt improvement service; not wired to auto-retry |
| `src/core/contextManager.ts` | Project context store with priority + anchor-point system |
| `src/task/taskManager.ts` | Task CRUD, Markdown persistence, LLM-powered generation |
| `src/index.ts` | MCP server setup (22 tools) + CLI yargs definition |
| `src/commands/helpImplementTaskHandler.ts` | Pair-programmer tool: most sophisticated LLM prompt in the codebase |
| `src/commands/researchTopicHandler.ts` | Research tool: Perplexity-first with provider fallback |
