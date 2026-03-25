# Architecture: Dual-Plane OpenClaw Routing System

## Overview

The OpenClaw routing system uses a **dual-plane architecture** that cleanly separates orchestration (what to do) from execution (how to do it). This document describes the design, state machine, routing contract, and operational details.

---

## Plane 1 — OpenClaw Orchestration

OpenClaw is the **control plane**. It owns:

- **Task intake** — receiving and parsing incoming coding requests
- **Dispatcher** — routing tasks to the external execution plane
- **Architect** — high-level design decisions and architecture review
- **Reviewer** — code review and quality gates
- **Designer** — UI/UX design decisions
- **Merge gate** — CI/CD integration and merge approval
- **Memory** — persistent knowledge and context across sessions
- **Session state** — conversation and workflow state management

OpenClaw does **NOT** own coding execution. All coding work is delegated to the ai-code-runner via a JSON stdin/stdout protocol.

---

## Plane 2 — External Execution (`ai-code-runner`)

`~/openclaw-router/` is the **data plane**. It owns:

- **Routing state machine** — determines which executor to use based on current state
- **Executor chain selection** — builds ordered fallback chains per task
- **Codex CLI adapter** — OpenAI native and OpenRouter backend integration
- **Claude Code adapter** — Anthropic backend integration
- **OpenRouter adapter** — dynamic model selection (minimax, kimi, etc.)
- **Fallback logic** — automatic retry and chain progression on failure
- **Structured logging** — JSONL event logging for observability

---

## File Layout

```
openclaw-router/
├── bin/
│   └── ai-code-runner              # stdin JSON entrypoint (main binary)
├── router/
│   ├── __init__.py                 # package exports
│   ├── models.py                   # TaskMeta, RouteDecision, ExecutorResult, ChainEntry
│   ├── classifier.py               # task classification logic
│   ├── policy.py                   # routing chain builder and state resolution
│   ├── executors.py                # Codex, Claude, OpenRouter adapters
│   ├── errors.py                   # normalized error types and mapping
│   ├── state_store.py              # manual + auto state management
│   ├── logger.py                   # JSONL structured logging
│   ├── flow_control.py             # multi-phase pipeline orchestration
│   └── output_format.py            # output format validation
├── config/
│   ├── router.config.json          # full config schema (routing, tools, retry)
│   ├── router.yaml                 # alternate config format
│   ├── codex_manual_state.json     # user-pinned state override
│   └── codex_auto_state.json       # system-tracked state (auto-recovered)
├── runtime/
│   ├── claude_health.json          # Claude Code health metrics
│   └── routing.jsonl               # append-only routing event log
└── tests/
    ├── test_architecture.py        # architecture validation tests
    ├── test_schemas.py             # model and config schema tests
    ├── test_classifier.py          # task classification tests
    ├── test_codex_executor.py      # Codex executor tests
    ├── test_claude_executor.py     # Claude executor tests
    ├── test_openrouter_executor.py # OpenRouter executor tests
    ├── test_state_store.py         # state management tests
    ├── test_errors.py              # error normalization tests
    ├── test_logger.py              # logging tests
    ├── test_flow_control.py        # pipeline flow tests
    ├── test_output_format.py       # output format tests
    ├── test_cost_tracking.py       # cost tracking tests
    └── test_integration.py         # integration tests
```

### Module Responsibilities

| Module | Purpose |
|---|---|
| `models.py` | Core data types: `TaskMeta` (task description), `RouteDecision` (routing plan), `ExecutorResult` (execution outcome), `ChainEntry` (chain step). Enums for task class, modality, risk, state, executor, backend, model profile. |
| `classifier.py` | Classifies incoming tasks by type, risk, and modality. Determines specialist routing needs (screenshots → kimi, etc.). |
| `policy.py` | Builds executor chains based on current state and task classification. Resolves state from manual/auto files. Manages fallback eligibility. |
| `executors.py` | Adapter layer for each backend: `run_codex()` (OpenAI native), `run_claude()` (Anthropic), `run_openrouter()` (dynamic model selection). Handles subprocess execution, timeout, and result normalization. |
| `errors.py` | Normalized error taxonomy: `auth_error`, `rate_limited`, `quota_exhausted`, `provider_unavailable`, `provider_timeout`, `transient_network_error`. Maps raw errors to normalized types. |
| `state_store.py` | Manages routing state via `codex_manual_state.json` (user-pinned) and `codex_auto_state.json` (system-tracked). Supports `normal` and `last10` states. |
| `logger.py` | Append-only JSONL logging to `runtime/routing.jsonl`. Records routing decisions, executor invocations, results, and errors. |
| `flow_control.py` | Multi-phase pipeline orchestration for complex tasks requiring sequential execution stages. |
| `output_format.py` | Validates and normalizes output formats for different task types. |

---

## State Machine

The router operates in one of two states, controlling which executor chain is used.

### States

| State | Description |
|---|---|
| `normal` | Default state. Uses full 3-step chain with Codex primary. |
| `last10` | Fallback state. Skips Codex primary, starts with Claude. |

### State Transitions

```
         ┌─────────────┐
         │   normal     │
         │ (default)    │
         └──────┬───────┘
                │ codex_cli failures (last 10 tasks)
                ▼
         ┌─────────────┐
         │   last10     │
         │ (fallback)   │
         └──────┬───────┘
                │ codex_cli recovery detected
                ▼
         ┌─────────────┐
         │   normal     │
         └─────────────┘
```

- **normal → last10**: Triggered when `codex_cli` fails on recent tasks (tracked in auto state)
- **last10 → normal**: Triggered when `codex_cli` recovers and succeeds
- **Manual override**: User can pin state via `codex_manual_state.json`

### Executor Chains by State

**Normal state** (3-step chain):
```
1. codex_cli + openai_native    (primary)
2. claude_code + anthropic      (secondary)
3. codex_cli + openrouter       (fallback)
```

**last10 state** (2-step chain):
```
1. claude_code + anthropic      (primary)
2. codex_cli + openrouter       (fallback)
```

---

## Supported Tool Backends

### 1. codex_cli + openai_native
- **Role**: Primary executor in normal state
- **Model**: `o3`
- **Backend**: OpenAI native API
- **Use case**: Standard implementation, refactor, bugfix, debug tasks

### 2. claude_code + anthropic
- **Role**: Secondary executor in normal state, primary in last10
- **Provider**: Anthropic
- **Use case**: Complex reasoning, code review, architecture-sensitive tasks

### 3. codex_cli + openrouter + minimax
- **Role**: Default open-source lane (fallback in both states)
- **Model**: Minimax via OpenRouter
- **Use case**: General fallback when primary/secondary fail

### 4. codex_cli + openrouter + kimi
- **Role**: Specialist executor
- **Model**: Kimi via OpenRouter
- **Use case**: Screenshots, multimodal tasks, swarm coordination
- **Triggered by**: `TaskClass.UI_FROM_SCREENSHOT`, `TaskClass.MULTIMODAL_CODE_TASK`, `TaskClass.SWARM_CODE_TASK`, or `has_screenshots=True`

### OpenRouter Dynamic Model Selection

When routing through `codex_cli + openrouter`, the model is selected dynamically:

- **Default**: `minimax` (open-source, cost-effective)
- **Specialist rules** (in `router.config.json`):
  - `ui_from_screenshot` → `kimi`
  - `multimodal_code_task` → `kimi`
  - `swarm_code_task` → `kimi`
  - `has_screenshots` → `kimi`

---

## Routing Contract

### Input: OpenClaw → ai-code-runner

The ai-code-runner receives a JSON object via stdin:

```json
{
  "task_meta": {
    "task_id": "task_001",
    "agent": "coder",
    "task_class": "implementation",
    "risk": "medium",
    "modality": "text",
    "requires_repo_write": true,
    "requires_multimodal": false,
    "has_screenshots": false,
    "swarm": false,
    "repo_path": "/path/to/repo",
    "cwd": "/path/to/repo",
    "summary": "Add login endpoint"
  },
  "prompt": "Implement a POST /login endpoint...",
  "context": {
    "files": ["app.py", "routes.py"],
    "recent_changes": "..."
  }
}
```

### Output: ai-code-runner → OpenClaw

The ai-code-runner returns an `ExecutorResult` via stdout:

```json
{
  "task_id": "task_001",
  "tool": "codex_cli",
  "backend": "openai_native",
  "model_profile": "codex_primary",
  "success": true,
  "normalized_error": null,
  "exit_code": 0,
  "latency_ms": 12340,
  "request_id": "req_abc123",
  "cost_estimate_usd": 0.045,
  "artifacts": ["app.py", "tests/test_login.py"],
  "stdout_ref": "runtime/outputs/task_001_stdout.txt",
  "stderr_ref": null,
  "final_summary": "Added POST /login endpoint with JWT auth, unit tests included."
}
```

### Key Contract Rules

- `RouteDecision` (routing plan) and `ExecutorResult` (execution outcome) are **separate types** sharing only `task_id`
- `RouteDecision` contains: `task_id`, `state`, `chain`, `reason`, `attempted_fallback`, `fallback_from`
- `ExecutorResult` contains: `task_id`, `tool`, `backend`, `model_profile`, `success`, `normalized_error`, `exit_code`, `latency_ms`, `request_id`, `cost_estimate_usd`, `artifacts`, `stdout_ref`, `stderr_ref`, `final_summary`
- No execution-time fields leak into `RouteDecision`
- No routing fields leak into `ExecutorResult`

---

## Error Handling and Fallback Chain

### Normalized Error Types

All executor errors are mapped to normalized types:

| Error Type | Description |
|---|---|
| `auth_error` | Invalid or expired API credentials |
| `rate_limited` | Provider rate limit hit |
| `quota_exhausted` | API quota or budget exceeded |
| `provider_unavailable` | Provider service down |
| `provider_timeout` | Provider did not respond in time |
| `transient_network_error` | Temporary network failure |

### Fallback Logic

1. Execute the first entry in the chain
2. If **success** → return `ExecutorResult` with `success=true`
3. If **failure** with eligible error → try next chain entry
4. If **failure** with non-eligible error → return immediately (no fallback)
5. If all chain entries exhausted → return `ExecutorResult` with `success=false`

### Retry Configuration

```json
{
  "max_retries": 2,
  "eligible_errors": [
    "auth_error",
    "rate_limited",
    "quota_exhausted",
    "provider_unavailable",
    "provider_timeout",
    "transient_network_error"
  ]
}
```

---

## Health Tracking

Claude Code health is tracked in `runtime/claude_health.json`:

- Consecutive success/failure counts
- Average latency
- Last error type
- Last successful execution timestamp

This data informs automatic state transitions (normal ↔ last10).

---

## Logging

All routing events are logged to `runtime/routing.jsonl` as append-only JSONL:

- **Routing decisions**: state, chain, reason
- **Executor invocations**: tool, backend, model, start time
- **Results**: success/failure, latency, cost, error type
- **State transitions**: old state → new state, trigger

Each log entry includes a timestamp and request ID for correlation.

---

## Design Principles

1. **Separation of concerns**: OpenClaw orchestrates, ai-code-runner executes
2. **Typed contracts**: `RouteDecision` and `ExecutorResult` are distinct types with no field leakage
3. **Graceful degradation**: Multi-step fallback chains with eligibility filtering
4. **Observability**: Structured JSONL logging for every routing decision
5. **State resilience**: Manual overrides survive restarts; auto state recovers from failures
6. **Dynamic model selection**: OpenRouter picks the right model based on task characteristics
