# Architecture: 4-State Subscription-Aware Routing System

## Overview

The OpenClaw routing system uses a **4-state subscription-aware architecture** that separates two independent control planes. The primary optimization is **subscription budget management** — use already-paid capacity (Codex, Claude subscriptions) before falling back to raw API usage (OpenRouter).

---

## Dual Control Planes

The routing system operates on two independent planes:

### Plane 1 — Subscription/Budget (Top-Level State)

Controls *which provider subscription bucket* to draw from. This is the primary routing dimension.

```
openai_primary → openai_conservation → claude_backup → openrouter_fallback
```

- **openai_primary**: Use the Codex/OpenAI subscription aggressively
- **openai_conservation**: Stay in OpenAI but conserve quota
- **claude_backup**: Fall back to Claude Code subscription
- **openrouter_fallback**: Pay-as-you-go via OpenRouter

### Plane 2 — Task/Capability (Model Within Lane)

Controls *which model within a provider* to use for a given task. This is the secondary dimension.

| Provider | Default | Hard/Expert | Specialist |
|----------|---------|-------------|------------|
| OpenAI | gpt-5.4-mini | gpt-5.4 | — |
| Claude | claude-sonnet-4.6 | claude-opus-4.6 | — |
| OpenRouter | minimax | mimo | kimi (visual) |

The two planes compose: subscription state determines *which lane to prefer*, task capability determines *which model within that lane*.

---

## State Machine

### States

| State | Goal | Primary Chain |
|-------|------|---------------|
| `openai_primary` | Maximize Codex subscription | Codex → Claude → OpenRouter |
| `openai_conservation` | Conserve OpenAI usage | Codex → OpenRouter → Claude |
| `claude_backup` | Use Claude subscription | Claude → OpenRouter |
| `openrouter_fallback` | Last resort | OpenRouter only |

### State Diagram

```
openai_primary
    ↓ (budget pressure: approaching weekly limit, rate warnings, manual switch)
openai_conservation
    ↓ (OpenAI exhausted, repeated rate_limited/quota_exhausted)
claude_backup
    ↓ (Claude unavailable, unauthed, disabled)
openrouter_fallback
```

### State Details

#### openai_primary (default)

- **Goal:** Maximize Codex subscription usage
- **Chain:** `codex_cli:openai_native` → `claude_code:anthropic` → `codex_cli:openrouter`
- **Model selection:** gpt-5.4 for critical/debug/architecture tasks, gpt-5.4-mini for everything else
- **When to use:** Default state, healthy OpenAI subscription

#### openai_conservation

- **Goal:** Stay in OpenAI lane but conserve subscription quota
- **Chain:** `codex_cli:openai_native` → `codex_cli:openrouter` → `claude_code:anthropic`
- **Model selection:** gpt-5.4-mini for almost everything, gpt-5.4 only for planner/final-review/high-risk
- **When to use:** Approaching weekly quota limit, rate pressure signals

#### claude_backup

- **Goal:** Use Claude Code subscription before paying OpenRouter
- **Chain:** `claude_code:anthropic` → `codex_cli:openrouter`
- **Model selection:** claude-sonnet-4.6 by default, claude-opus-4.6 for hardest cases
- **When to use:** OpenAI exhausted, repeated rate_limited/quota_exhausted

#### openrouter_fallback

- **Goal:** Last resort paid usage
- **Chain:** `codex_cli:openrouter` only
- **Model selection:** minimax default, mimo for hardest tasks, kimi for visual/multimodal
- **When to use:** Claude unavailable, unauthed, or disabled

### State Transitions

| From | To | Trigger |
|------|----|---------|
| openai_primary | openai_conservation | Manual switch, approaching quota, rate pressure |
| openai_conservation | claude_backup | OpenAI exhausted, repeated rate_limited/quota_exhausted |
| claude_backup | openrouter_fallback | Claude unavailable, unauthed, or disabled |
| Any | Any | Hard failure override (auth error, provider outage) |

---

## Model Routing Within Each State

### OpenAI Lane

- **gpt-5.4**: Used for `risk=critical`, `task_class=debug`, `task_class=repo_architecture_change`, planner, final-review
- **gpt-5.4-mini**: Everything else — code search, reading, docs, simple fixes, standard implementation
- In `openai_conservation`, gpt-5.4 usage is restricted to highest-priority tasks only

### Claude Lane

- **claude-sonnet-4.6**: Default Claude executor — good quality, reasonable cost
- **claude-opus-4.6**: Reserved for hardest cases — complex architecture, multi-file refactors, critical decisions

### OpenRouter Lane

- **minimax** (`minimax/minimax-m2.7`): Default fallback — broad compatibility, low cost
- **mimo** (`xiaomi/mimo-v2-pro`): Hardest tasks, 1M context window, orchestrator brain in `openrouter_fallback`
- **kimi** (`moonshotai/kimi-k2.5`): Visual/multimodal specialist — screenshots, image analysis

---

## Budget Heuristics

### Manual Override (Primary Control)

The user can manually switch states at any time via:

```bash
echo '{"state": "openai_conservation"}' > config/codex_manual_state.json
```

Manual state takes precedence over all automatic signals. It persists across restarts until cleared.

### Signal-Based Transitions

The router monitors execution results for budget-related signals:

- **rate_limited**: Provider returned 429 or rate limit error
- **quota_exhausted**: Provider returned 402 or quota exceeded error
- **auth_error**: Authentication failure (may indicate subscription issue)

When these signals accumulate, the router can auto-transition to a more conservative state.

### Local Soft Counting

The router maintains local counters for rough budget tracking:

- Task count per provider (rolling window)
- Rough token usage estimates
- Time-since-last-success tracking

These are **soft** signals — they inform transitions but don't trigger them alone. Hard signals (rate_limited, quota_exhausted) are the primary triggers.

### Sticky State

State transitions are **sticky** — the router doesn't bounce back to a higher tier after a single success. To return to `openai_primary` from `openai_conservation`:

1. Manual override, OR
2. Sustained success over a cooldown period (configurable)

This prevents rapid state oscillation during partial outages.

---

## Fallback Edges

Failures during task execution are **secondary overrides**, not the main routing spine:

1. Execute the first entry in the chain
2. If **success** → return result
3. If **fallback-eligible error** → try next chain entry
4. If **non-eligible error** → return immediately (no fallback)
5. If all chain entries exhausted → return failure

### Fallback-Eligible Errors

| Error | Description |
|-------|-------------|
| `auth_error` | Authentication or credential failure |
| `rate_limited` | Provider rate limit hit |
| `quota_exhausted` | API quota or billing exhausted |
| `provider_unavailable` | Provider service down |
| `provider_timeout` | Provider response timeout |
| `transient_network_error` | Temporary network failure |

### Non-Eligible Errors (Stop Immediately)

| Error | Description |
|-------|-------------|
| `invalid_payload` | Malformed task JSON |
| `missing_repo_path` | Required repo path not provided |
| `permission_denied_local` | Local file permission error |
| `git_conflict` | Git merge/rebase conflict |
| `toolchain_error` | Missing compiler, linter, or toolchain |
| `template_render_error` | Code template rendering failure |
| `unsupported_task` | Task type not supported by executor |

---

## Health Tracking

### Claude Health

Claude Code availability is tracked in `runtime/claude_health.json`:

- Consecutive success/failure counts
- Average latency
- Last error type
- Last successful execution timestamp

When Claude health degrades, the router can auto-transition from `claude_backup` to `openrouter_fallback`.

### OpenAI Budget Signals

OpenAI usage is tracked via:

- Task count per state (rolling window)
- Rate limit hit frequency
- Quota exhaustion signals

These inform transitions from `openai_primary` to `openai_conservation` and beyond.

---

### Observability

| Feature | Implementation | Location |
|---------|---------------|----------|
| Trace ID | 12-char hex uuid4 per route decision | `router/policy.py:route_task()` |
| Structured logging | routing_trace JSONL per decision | `router/attempt_logger.py` |
| Metrics aggregation | MetricsCollector from JSONL | `router/metrics.py` |
| Circuit breaker logging | providers_skipped in trace | `router/circuit_breaker.py` |
| State history | codex_state_history.json | `router/state_store.py` |

### State Management

| Feature | Implementation |
|---------|---------------|
| Atomic writes | tempfile + os.replace + fsync |
| WAL/Journal | Append-only JSONL before state writes, committed marker after |
| Sticky state | N=3 consecutive successes required before automated recovery to primary |
| Chain invariants | validate_chain() — claude_backup no openai_native, openrouter_fallback only openrouter |
| Anti-flap | 300s minimum between transitions |
| Emergency override | OPENAI_PRIMARY, CLAUDE_BACKUP always allowed |
| State history | Last 50 transitions with timestamps |

---

## Routing Contract

### Input: OpenClaw → ai-code-runner

The ai-code-runner receives a JSON object via stdin containing `task_meta` with fields: `task_id`, `agent`, `task_class`, `risk`, `modality`, `requires_repo_write`, `requires_multimodal`, `has_screenshots`, `swarm`, `repo_path`, `cwd`, `summary`.

### Output: ai-code-runner → OpenClaw

The ai-code-runner returns an `ExecutorResult` via stdout with fields: `task_id`, `tool`, `backend`, `model_profile`, `success`, `normalized_error`, `exit_code`, `latency_ms`, `request_id`, `cost_estimate_usd`, `artifacts`, `stdout_ref`, `stderr_ref`, `final_summary`.

### Key Contract Rules

- `RouteDecision` (routing plan) and `ExecutorResult` (execution outcome) are **separate types** sharing only `task_id`
- No execution-time fields leak into `RouteDecision`
- No routing fields leak into `ExecutorResult`

---

## Supported Tool Backends

### 1. codex_cli + openai_native (Backend: OpenAI)
- **Role**: Primary executor in openai_primary and openai_conservation
- **Models**: gpt-5.4 (heavy), gpt-5.4-mini (light)
- **Backend**: OpenAI native API via Codex CLI

### 2. claude_code + anthropic (Backend: Anthropic)
- **Role**: Secondary in openai_primary, primary in claude_backup
- **Models**: claude-sonnet-4.6 (default), claude-opus-4.6 (hard)
- **Backend**: Anthropic API via Claude Code

### 3. codex_cli + openrouter (Backend: OpenRouter)
- **Role**: Fallback in all states, only executor in openrouter_fallback
- **Models**: minimax (default), mimo (hard/orchestrator), kimi (visual)
- **Backend**: OpenRouter API

### Migration Note

The system previously used 2 states (`normal` and `last10`) driven by failure history. The new 4-state architecture (`openai_primary`, `openai_conservation`, `claude_backup`, `openrouter_fallback`) is driven by subscription budget management. The `normal` state mapped to `openai_primary` and `last10` mapped to a simplified `claude_backup` pattern.

---

## File Layout

| Module | Purpose |
|--------|---------|
| `models.py` | Core data types: `TaskMeta`, `RouteDecision`, `ExecutorResult`, `ChainEntry`. Enums for task class, modality, risk, state, executor, backend, model profile. |
| `classifier.py` | Classifies incoming tasks by type, risk, and modality. Determines specialist routing needs (screenshots → kimi, etc.). |
| `policy.py` | Routing policy — 4-state chain builders, model selection per lane. Resolves state from manual/auto files. |
| `executors.py` | Adapter layer for each backend: `run_codex()`, `run_claude()`, `run_openrouter()`. Handles subprocess execution, timeout, and result normalization. |
| `errors.py` | Normalized error taxonomy and fallback eligibility mapping. |
| `state_store.py` | Persistent state store — manual/auto layers, budget signal tracking. Manages `codex_manual_state.json` and `codex_auto_state.json`. |
| `logger.py` | Append-only JSONL logging to `runtime/routing.jsonl`. |
| `config_loader.py` | `get_model()` — config-driven model strings from `router.config.json`. |
| `flow_control.py` | Multi-phase pipeline orchestration for complex tasks. |
| `output_format.py` | Output format validation and normalization. |

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

1. **Subscription-first**: Use paid subscriptions before raw API spend
2. **Separation of concerns**: OpenClaw orchestrates, ai-code-runner executes
3. **Typed contracts**: `RouteDecision` and `ExecutorResult` are distinct types
4. **Graceful degradation**: Multi-step fallback chains with eligibility filtering
5. **Observability**: Structured JSONL logging for every routing decision
6. **State resilience**: Manual overrides survive restarts; auto state recovers from failures
7. **Dynamic model selection**: Right model for each task within each lane
