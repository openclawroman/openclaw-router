# ai-code-runner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1784%20passing-brightgreen.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-95.73%25-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**An external routing layer for OpenClaw coding tasks.** Classifies tasks, selects the right executor (Codex CLI, Claude Code, or OpenRouter), handles fallback, and logs everything.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Dual-Plane Design](#dual-plane-design)
  - [State Machine](#state-machine)
- [Executors](#executors)
- [Contract Objects](#contract-objects)
  - [TaskMeta (Input)](#taskmeta-input)
  - [RouteDecision (Routing Output)](#routedecision-routing-output)
  - [ExecutorResult (Execution Output)](#executorresult-execution-output)
- [Error Handling](#error-handling)
- [Reviewer Independence](#reviewer-independence)
- [File Layout](#file-layout)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [Benchmark](#benchmark)
- [Security & Privacy](#security--privacy)
- [Lifecycle & Maintenance](#lifecycle--maintenance)
- [Monitoring](#monitoring)
- [Documentation](#documentation)

---

## Overview

`ai-code-runner` is the **data plane** for OpenClaw's coding execution. It receives structured JSON tasks, classifies them by type and risk, builds an executor chain with automatic fallback, runs the task through the selected coding tool, and returns a structured result.

**Key properties:**

- **Zero external dependencies** — pure Python 3.10+ stdlib (only `pytest` for tests)
- **JSON in, JSON out** — pipe-friendly `stdin`/`stdout` protocol
- **Subscription-aware routing** — 4-state system that maximizes paid subscription buckets before falling back to raw API usage
- **Structured logging** — every routing decision logged to JSONL with per-executor attempt trail
- **Trace ID correlation** — every routing decision has a unique 12-char trace ID for log correlation across fallback attempts
- **Metrics aggregation** — built-in metrics collector for task distribution, model usage, fallback rates, and cost per provider
- **Atomic state writes** — crash-safe state file updates with history tracking and anti-flap protection
- **WAL/Journal** — write-ahead log for crash recovery of state changes
- **Sticky state** — requires N consecutive successes before automated recovery from degraded state
- **Chain invariants** — validates routing chains respect state-specific provider rules
- **Provider weights** — composite scoring (reliability×0.4 + speed×0.2 + cost×0.2 + priority×0.2)
- **Rate limit parsing** — extracts rate limit headers from OpenAI, Anthropic, OpenRouter responses
- **Notifications** — state change alerts, fallback rate warnings, conservation duration alerts
- **Provider dashboard** — `--status` CLI for real-time provider health summary
- **Audit chain** — tamper-evident SHA-256 chain of routing decisions for full traceability
- **Secret redaction** — automatic detection and masking of API keys, tokens, and credentials in logs
- **Content isolation** — task content isolated from cross-task contamination
- **File permissions** — state files written with restrictive `0o600` permissions

---

## Architecture

### Dual-Plane Design

The routing system uses a clean **two-plane architecture** that separates orchestration from execution:

| Plane | Owner | Responsibility |
|-------|-------|----------------|
| **Plane 1 — Orchestration** | OpenClaw | Task intake, dispatch, architecture review, code review, design decisions, merge gates, memory, session state |
| **Plane 2 — Execution** | ai-code-runner | Routing state machine, executor chain selection, Codex CLI, Claude Code, OpenRouter, fallback logic, structured logging |

OpenClaw controls **what** to do. `ai-code-runner` controls **how** to do it.

> 📖 Full details in [Architecture Spec](docs/architecture.md)

### State Machine

The router uses a 4-state subscription-aware architecture. The primary state machine optimizes for **subscription buckets** — use already-paid capacity before paying for raw API usage.

```
openai_primary
    ↓ (budget pressure: approaching weekly limit, rate warnings, manual switch)
openai_conservation
    ↓ (OpenAI exhausted, repeated rate_limited/quota_exhausted)
claude_backup
    ↓ (Claude unavailable, unauthed, disabled)
openrouter_fallback
```

#### openai_primary (default)
**Goal:** Maximize Codex subscription usage.
**Chain:** Codex CLI → Claude Code → OpenRouter
**Model selection:** gpt-5.4 for critical/debug/architecture, gpt-5.4-mini for everything else.

#### openai_conservation
**Goal:** Stay in OpenAI lane but conserve subscription quota.
**Chain:** Codex CLI → Claude Code → OpenRouter
**Model selection:** gpt-5.4-mini for almost everything, gpt-5.4 only for planner/final-review/high-risk.

#### claude_backup
**Goal:** Use Claude Code subscription before paying OpenRouter.
**Chain:** Claude Code → OpenRouter
**Model selection:** claude-sonnet-4.6 by default, claude-opus-4.6 for hardest cases.

#### openrouter_fallback
**Goal:** Last resort paid usage.
**Chain:** OpenRouter only
**Model selection:** minimax default, mimo for hardest tasks, kimi for visual/multimodal.

#### State Transitions

| From | To | Trigger |
|------|----|---------|
| openai_primary | openai_conservation | Manual switch, approaching quota, rate pressure |
| openai_conservation | claude_backup | OpenAI exhausted, repeated rate_limited/quota_exhausted |
| claude_backup | openrouter_fallback | Claude unavailable, unauthed, or disabled |
| Any | Any | Hard failure override (auth error, provider outage) |

---

## Executors

| # | Tool | Backend | Model/Profile | State | Role |
|---|------|---------|---------------|-------|------|
| 1a | `codex_cli` | `openai_native` | **gpt-5.4** | openai_primary | **Heavy lane** — planning, hard debugging, risky work, final judgment |
| 1b | `codex_cli` | `openai_native` | **gpt-5.4-mini** | openai_primary/conservation | **Light lane** — code search, reading, docs, simple fixes |
| 2a | `claude_code` | `anthropic` | **claude-sonnet-4.6** | claude_backup | **Backup lane** — default Claude coding executor |
| 2b | `claude_code` | `anthropic` | **claude-opus-4.6** | claude_backup | **Hard Claude** — critical/architecture when Claude is primary |
| 3 | `codex_cli` | `openrouter` | minimax (config-driven) | any state | **Default fallback** — broad compatibility, low cost |
| 4 | `codex_cli` | `openrouter` | mimo (config-driven) | openrouter_fallback | **Orchestrator brain** — hardest tasks, 1M context |
| 5 | `codex_cli` | `openrouter` | kimi (config-driven) | openrouter_fallback | **Visual specialist** — screenshots, image analysis |

> **Model strings are config-driven** — all model names live in `config/router.config.json` under `models`. To swap a model, edit one line there. No code changes needed. See [Changing Models](#changing-models).

Executor selection depends on:

- Current router state (see [State Machine](#state-machine))
- Task modality (`text`, `image`, `video`, `mixed`)
- Task flags (`has_screenshots`, `requires_multimodal`, `swarm`)
- Subscription budget and rate signals

### Changing Models

All model strings live in `config/router.config.json`. One line, one file.

```json
{
  "models": {
    "openrouter": {
      "minimax": "minimax/minimax-m2.7",
      "kimi": "moonshotai/kimi-k2.5",
      "mimo": "xiaomi/mimo-v2-pro"
    },
    "codex": {
      "gpt54": "gpt-5.4",
      "gpt54_mini": "gpt-5.4-mini"
    },
    "claude": {
      "sonnet": "claude-sonnet-4.6",
      "opus": "claude-opus-4.6"
    }
  }
}
```

No code changes. No restart needed.

---

## Contract Objects

All data flows through three typed objects:

### TaskMeta (Input)

Sent to `ai-code-runner` via `stdin` JSON. Describes what needs to be done.

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `task_id` | `str` | — | Unique task identifier |
| `agent` | `str` | `coder`, `reviewer`, `architect`, `designer`, `worker` | Agent role requesting execution |
| `task_class` | `str` | 10 types (e.g. `implementation`, `refactor`, `test`, `review`, ...) | Task classification |
| `risk` | `str` | `low`, `medium`, `high`, `critical` | Risk level — affects review depth |
| `modality` | `str` | `text`, `image`, `video`, `mixed` | Content modality |
| `requires_repo_write` | `bool` | — | Whether the task writes to the repo |
| `requires_multimodal` | `bool` | — | Whether multimodal capabilities needed |
| `has_screenshots` | `bool` | — | Whether screenshots are involved |
| `swarm` | `bool` | — | Whether this is a swarm task |
| `repo_path` | `str` | — | Path to the target repository |
| `cwd` | `str` | — | Working directory |
| `summary` | `str` | — | Human-readable task description |

### RouteDecision (Routing Output)

Generated after routing and logged to `runtime/routing.jsonl`. Describes the executor chain and routing rationale. **Not returned on stdout** — only available in the JSONL log file.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Task identifier |
| `state` | `str` | Router state: `openai_primary`, `openai_conservation`, `claude_backup`, or `openrouter_fallback` |
| `chain` | `list[ChainEntry]` | Ordered list of executor entries to try |
| `reason` | `str` | Human-readable routing explanation |
| `attempted_fallback` | `bool` | Whether fallback was triggered |
| `fallback_from` | `str | None` | Which executor failed, triggering fallback |

### ExecutorResult (Execution Output)

Returned after execution. Contains the outcome, metrics, and artifacts.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Task identifier |
| `tool` | `str` | Executor tool used (`codex_cli`, `claude_code`) |
| `backend` | `str` | Backend used (`openai_native`, `anthropic`, `openrouter`) |
| `model_profile` | `str | None` | Model profile (e.g. `openrouter_minimax`) |
| `success` | `bool` | Whether execution succeeded |
| `normalized_error` | `str | None` | Error category (see [Error Handling](#error-handling)) |
| `exit_code` | `int | None` | Process exit code |
| `latency_ms` | `float | None` | Execution latency in milliseconds |
| `request_id` | `str | None` | Provider request ID for tracing |
| `cost_estimate_usd` | `float | None` | Estimated cost in USD |
| `artifacts` | `list[str] | None` | Generated file paths |
| `stdout_ref` | `str | None` | Reference to stdout output |
| `stderr_ref` | `str | None` | Reference to stderr output |
| `final_summary` | `str | None` | Human-readable result summary |

---

## Error Handling

Errors are categorized into two groups based on fallback eligibility:

### ✅ Fallback-Eligible Errors

These trigger the next executor in the chain:

| Error | Description |
|-------|-------------|
| `auth_error` | Authentication or credential failure |
| `rate_limited` | Provider rate limit hit |
| `quota_exhausted` | API quota or billing exhausted |
| `provider_unavailable` | Provider service down |
| `provider_timeout` | Provider response timeout |
| `transient_network_error` | Temporary network failure |

### ❌ Non-Eligible Errors (Fail Immediately)

These stop execution — no fallback is attempted:

| Error | Description |
|-------|-------------|
| `invalid_payload` | Malformed task JSON |
| `missing_repo_path` | Required repo path not provided |
| `permission_denied_local` | Local file permission error |
| `git_conflict` | Git merge/rebase conflict |
| `toolchain_error` | Missing compiler, linter, or toolchain |
| `template_render_error` | Code template rendering failure |
| `unsupported_task` | Task type not supported by executor |
| `context_too_long` | Input exceeds context window limits |
| `content_filtered` | Content blocked by provider safety/policy filter |

### ⚠️ Partial Success

When an executor produces useful output before failing (e.g., it wrote files but exited non-zero), the result is marked `partial_success=True`. In this case the chain **stops immediately** — no fallback is attempted — and the partial result is returned to the caller.

---

## Reviewer Independence

To ensure objective code review, the system enforces **reviewer independence**:

- **Code generation and review must use different executors** — the tool that generated code cannot review it
- **Review depth scales with risk:**
  - `FAST` (default) — quick review for low/medium risk
  - `DEEP` — thorough review for high/critical risk or architecture changes

### Merge Gate

`merge_gate()` validates before any merge is allowed:

- ✅ Tests pass
- ✅ Lint passes
- ✅ No sensitive files touched (secrets, credentials, `.env`)
- ✅ Reviewer independence confirmed (different executor than generator)

---

## File Layout

```
openclaw-router/
├── bin/ai-code-runner          # stdin JSON → stdout JSON entrypoint
├── router/
│   ├── __init__.py             # Package exports
│   ├── models.py               # TaskMeta, RouteDecision, ExecutorResult, ChainEntry
│   ├── classifier.py           # classify() — keyword-based task classification
│   ├── policy.py               # Routing policy — 4-state chain builders, model selection per lane
│   ├── executors.py            # run_codex(), run_claude(), run_openrouter()
│   ├── errors.py               # normalize_error(), error classes, fallback eligibility
│   ├── state_store.py          # Persistent state store — WAL, sticky state, anti-flap
│   ├── attempt_logger.py       # Structured routing trace with per-executor attempt trail
│   ├── metrics.py              # Aggregated metrics (task distribution, cost, fallback rates)
│   ├── notifications.py        # State change alerts, conservation/fallback warnings
│   ├── rate_limits.py          # Rate limit header parsing (OpenAI/Anthropic/OpenRouter)
│   ├── provider_weights.py     # Composite scoring for cost/reliability-aware routing
│   ├── provider_dashboard.py   # Health summary (--status CLI)
│   ├── circuit_breaker.py      # Per-provider circuit breaker with health states
│   ├── health.py               # Health check endpoint, graceful shutdown
│   ├── config_loader.py        # get_model() — config-driven model strings
│   ├── config_validator.py     # Config validation at startup
│   ├── config_migration.py     # Config schema migration between versions
│   ├── environments.py         # Environment-specific config overrides
│   ├── logger.py               # RoutingLogger — JSONL event logging
│   ├── telemetry.py            # RouteQualityReporter — routing quality metrics
│   ├── output_format.py        # Output format validation
│   ├── flow_control.py         # Pipeline phase management
│   ├── audit.py                # Tamper-evident SHA-256 audit chain
│   ├── sanitize.py             # Input/output sanitization
│   ├── secrets.py              # Secret detection and redaction
│   ├── model_registry.py       # Model versioning registry
│   └── benchmark.py            # Routing latency benchmark harness
├── config/
│   ├── router.config.json      # Full configuration schema
│   ├── router.yaml             # Model configuration
│   ├── codex_manual_state.json # Manual state override
│   ├── codex_auto_state.json   # Auto state (managed by router)
│   ├── codex_state_wal.jsonl   # Write-ahead log for state persistence
│   └── codex_state_history.json # State transition history
├── runtime/
│   ├── routing.jsonl           # Append-only routing decision log
│   └── alerts.jsonl            # Notification alert log
├── docs/
│   ├── architecture.md         # Full architecture spec
│   └── runbooks.md             # Operational runbooks
├── tests/
│   └── (71 test files, 1784 tests)
└── README.md                   # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- POSIX-compatible shell (bash, zsh)

### Clone & Install

```bash
git clone https://github.com/openclawroman/openclaw-router.git
cd openclaw-router

# No external dependencies needed for runtime
# Install pytest for testing:
pip3 install pytest
```

### Verify

```bash
python3 -m pytest tests/ -q
```

Expected: `1784 passed`

---

## Usage

### Run a Task

Pipe a JSON task through the entrypoint:

```bash
echo '{
  "task_meta": {
    "task_id": "test-001",
    "agent": "coder",
    "task_class": "implementation",
    "risk": "medium",
    "modality": "text",
    "requires_repo_write": false,
    "requires_multimodal": false,
    "has_screenshots": false,
    "swarm": false,
    "repo_path": "/tmp",
    "cwd": "/tmp",
    "summary": "say hello"
  }
}' | python3 bin/ai-code-runner
```

The output is a JSON object (`ExecutorResult`) on stdout. The `RouteDecision` (routing rationale, executor chain, state) is logged to `runtime/routing.jsonl` for audit/traceability.

> **⚠️ `cwd` is required:** Codex exec (`codex exec [prompt]`) needs a working directory. If `cwd` is not set, codex fails with `toolchain_error`. Always set `cwd` in `task_meta` (or `repo_path` as fallback).

### Run Tests

```bash
# Full suite
pytest tests/ -v

# Quick run
pytest tests/ -q

# Specific test file
pytest tests/test_classifier.py -v
```

---

## Benchmark

The router ships with a built-in latency benchmark harness (`router/benchmark.py`) that measures end-to-end routing overhead.

```bash
# Run benchmark tests
pytest tests/test_routing_benchmark.py -v

# Direct benchmark invocation
python3 -m router.benchmark
```

The benchmark exercises classifier throughput, policy evaluation speed, state store read/write latency, and full pipeline round-trip time. Results are emitted as structured JSON suitable for CI regression tracking.

---

## Security & Privacy

`ai-code-runner` implements defense-in-depth for sensitive data:

- **Secret redaction** — API keys, tokens, and credentials are automatically detected and masked in all log output before they reach the filesystem. See `router/secrets.py`.
- **Content isolation** — task content is isolated per-execution; no cross-task contamination between routing decisions.
- **File permissions** — all state files (`codex_auto_state.json`, WAL, history) are written with `0o600` permissions (owner read/write only).
- **Audit chain integrity** — every routing decision is appended to a tamper-evident SHA-256 chain (`router/audit.py`). Any modification to a historical entry invalidates the entire chain.

---

## Lifecycle & Maintenance

- **Model versioning** — the model registry (`router/model_registry.py`) tracks available model versions, allowing safe deprecation and promotion without code changes.
- **Config migration** — `router/config_migration.py` handles schema upgrades when config formats evolve between releases. Existing configs are automatically migrated at startup.

---

## Monitoring

### Live Routing Log

```bash
tail -f runtime/routing.jsonl | python3 -m json.tool
```

### Health Check

The router tracks Claude Code availability in `runtime/claude_health.json`. If Claude is down, the router automatically adjusts the executor chain.

### State Inspection

```bash
# Current auto state
cat config/codex_auto_state.json | python3 -m json.tool

# Manual override (if set)
cat config/codex_manual_state.json | python3 -m json.tool

# Current state via Python
python3 -c "from router.policy import resolve_state; print(f'State: {resolve_state().value}')"
```

Valid states: `openai_primary`, `openai_conservation`, `claude_backup`, `openrouter_fallback`

> 📖 Full operational guides in [Runbooks](docs/runbooks.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Spec](docs/architecture.md) | Full dual-plane design, state machine details, routing contract, and data flow |
| [Operational Runbooks](docs/runbooks.md) | Installation, state override, provider setup, monitoring, dry runs |

---

## License

MIT
