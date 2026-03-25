# ai-code-runner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-534%20passing-brightgreen.svg)](#testing)
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
- [Monitoring](#monitoring)
- [Documentation](#documentation)

---

## Overview

`ai-code-runner` is the **data plane** for OpenClaw's coding execution. It receives structured JSON tasks, classifies them by type and risk, builds an executor chain with automatic fallback, runs the task through the selected coding tool, and returns a structured result.

**Key properties:**

- **Zero external dependencies** — pure Python 3.10+ stdlib (only `pytest` for tests)
- **JSON in, JSON out** — pipe-friendly `stdin`/`stdout` protocol
- **Stateful routing** — adapts executor chain based on recent failure history
- **Structured logging** — every routing decision logged to JSONL

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

The router operates in two states based on recent execution history:

| State | Chain (in priority order) | Condition |
|-------|---------------------------|-----------|
| **normal** | `codex_cli:openai_native` → `claude_code:anthropic` → `codex_cli:openrouter` | Default — Codex-native is healthy |
| **last10** | `claude_code:anthropic` → `codex_cli:openrouter` | Codex-native had failures in last 10 runs — skip it |

**Transitions:**

- **normal → last10**: Triggered when `codex_cli:openai_native` hits a fallback-eligible error threshold
- **last10 → normal**: Triggered after Claude Code succeeds and a cooldown period elapses

---

## Executors

| # | Tool | Backend | Model/Profile | Role |
|---|------|---------|---------------|------|
| 1a | `codex_cli` | `openai_native` | **gpt-5.4** | **Heavy lane** — planning, hard debugging, risky multi-file work, final judgment |
| 1b | `codex_cli` | `openai_native` | **gpt-5.4-mini** | **Light lane** — code search, reading files, docs, simple fixes, cheap tasks |
| 2 | `claude_code` | `anthropic` | Claude | **Secondary** in normal, **primary** in last10 — high-quality code generation |
| 3 | `codex_cli` | `openrouter` | minimax (config-driven) | **Open-source lane** — default fallback for broad compatibility |
| 4 | `codex_cli` | `openrouter` | kimi (config-driven) | **Multimodal specialist** — screenshots, image analysis, swarm tasks |

> **OpenAI profile selection** — `choose_openai_profile(task)` picks gpt-5.4 when `risk=critical`, `task_class=debug`, or `task_class=repo_architecture_change`. Everything else gets gpt-5.4-mini.

> **Model strings are config-driven** — all model names live in `config/router.config.json` under `models`. To swap a model, edit one line there. No code changes needed. See [Changing Models](#changing-models).

Executor selection depends on:

- Current router state (`normal` vs `last10`)
- Task modality (`text`, `image`, `video`, `mixed`)
- Task flags (`has_screenshots`, `requires_multimodal`, `swarm`)
- Previous failures in the chain

### Changing Models

All model strings live in **one place**: `config/router.config.json` under the `models` section.

```json
{
  "models": {
    "openrouter": {
      "minimax": "minimax/minimax-m2.7",
      "kimi": "moonshotai/kimi-k2.5"
    },
    "codex": {
      "gpt54": "gpt-5.4",
      "gpt54_mini": "gpt-5.4-mini"
    }
  }
}
```

To swap a model (e.g. when a new version releases), edit **one line** in this file. Both `policy.py` and `executors.py` read models via `get_model()` from `router/config_loader.py` — no code changes needed.

```bash
# Example: upgrade minimax to v3.0
# Edit config/router.config.json:
#   "minimax": "minimax/minimax-m3.0"
# Done.
```

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

Returned after routing. Describes the executor chain and routing rationale.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Task identifier |
| `state` | `str` | Router state: `normal` or `last10` |
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
│   ├── policy.py               # route_task(), build_chain(), get_review_chain(), merge_gate()
│   ├── executors.py            # run_codex(), run_claude(), run_openrouter()
│   ├── errors.py               # normalize_error(), error classes, fallback eligibility
│   ├── state_store.py          # StateStore — manual/auto state files
│   ├── logger.py               # RoutingLogger — JSONL event logging
│   ├── telemetry.py            # RouteQualityReporter — routing quality metrics
│   ├── config_loader.py        # get_model() — config-driven model strings
│   ├── output_format.py        # Output format validation
│   └── flow_control.py         # Pipeline phase management
├── config/
│   ├── router.config.json      # Full configuration schema
│   ├── router.yaml             # Model configuration
│   ├── codex_manual_state.json # Manual state override
│   └── codex_auto_state.json   # Auto state (managed by router)
├── runtime/
│   ├── claude_health.json      # Claude availability tracking
│   └── routing.jsonl           # Append-only routing decision log
├── docs/
│   ├── architecture.md         # Full architecture spec
│   └── runbooks.md             # Operational runbooks
├── tests/
│   └── (14 test files, 534 tests)
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

Expected: `534 passed`

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
    "summary": "say hello"
  }
}' | python3 bin/ai-code-runner
```

The output is a JSON object containing the `RouteDecision` and `ExecutorResult`.

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
```

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
