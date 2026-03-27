# BRIDGE_CONTRACT.md — OpenClaw Router Bridge Contract

Stable public contract for bridge authors integrating with the OpenClaw router.

## Input Schema

The router accepts a JSON object on stdin:

```json
{
    "protocol_version": 1,
    "task_meta": {
        "task_id": "string (required)",
        "agent": "string (default: 'coder') — coder|reviewer|architect|designer|worker",
        "task_class": "string (default: 'implementation') — implementation|review|planner|final_review|ui_from_screenshot|multimodal_code_task",
        "risk": "string (default: 'medium') — low|medium|high",
        "modality": "string (default: 'text') — text|multimodal",
        "requires_repo_write": "bool (default: false)",
        "requires_multimodal": "bool (default: false)",
        "has_screenshots": "bool (default: false)",
        "swarm": "bool (default: false)",
        "repo_path": "string",
        "cwd": "string — working directory override",
        "summary": "string — task summary (auto-derived from prompt if omitted)"
    },
    "prompt": "string — the user's task prompt",
    "attachments": [
        {"path": "/absolute/path/to/file", "type": "screenshot|file|image"}
    ],
    "scope": {
        "scope_id": "string",
        "thread_id": "string",
        "session_id": "string"
    },
    "context": {
        "branch": "string — git branch alias (shorthand)",
        "cwd": "string — working directory",
        "git_branch": "string — git branch name",
        "language": "string — primary programming language",
        "task_type": "string — task type hint",
        "model_preferences": {"key": "value pairs — provider/model preferences"},
        "working_directory": "string — alternative cwd key",
        "target_files": ["list of file paths to focus on"]
    },
    "continuity": {
        "previous_task_id": "string",
        "conversation_turns": "int"
    }
}
```

### Field Details

| Field | Required | Description |
|-------|----------|-------------|
| `protocol_version` | No (default: 1) | Integer version for the bridge protocol. Echoed in output. |
| `task_meta` | Yes | Core task metadata — at minimum `task_id`. |
| `prompt` | No | Full task prompt. Used to auto-derive `summary` if omitted. |
| `attachments` | No | File attachments. Each must have `path` (absolute). |
| `scope` | No | Routing scope identifiers (scope_id, thread_id, session_id). |
| `context` | No | Execution context — working dir, git branch, language, model prefs. |
| `continuity` | No | Conversation continuity metadata for multi-turn sessions. |

## Output Schema

The router emits a JSON object on stdout:

```json
{
    "protocol_version": 1,
    "task_id": "string",
    "tool": "string — executor tool used (e.g. codex_cli, claude_cli)",
    "backend": "string — backend provider (e.g. openai_native, anthropic_native)",
    "model_profile": "string — model profile name",
    "success": "bool",
    "normalized_error": "string|null — machine-readable error code",
    "exit_code": "int|null — executor process exit code",
    "latency_ms": "int — total wall-clock latency",
    "request_id": "string|null — provider request ID",
    "cost_estimate_usd": "float|null — estimated cost",
    "artifacts": ["list of artifact file paths"],
    "stdout_ref": "string|null — path to stdout log file",
    "stderr_ref": "string|null — path to stderr log file",
    "final_summary": "string|null — human-readable summary of what was done"
}
```

### Error Output

On errors (parse failures, routing errors), the router emits to stderr:

```json
{"error": "description", "type": "ErrorClassName", "protocol_version": 1}
```

## Versioning

- `protocol_version` is an integer, currently `1`.
- It is set on input and echoed unchanged on output.
- Bridge authors should set this field. The router defaults to `1` if omitted.
- Future protocol changes will bump this version. Backward-compatible additions (new optional fields) do NOT bump the version.

## Environment Variables Set for Executors

The router sets the following env vars before invoking executors. Executor tools (codex_cli, claude_cli, etc.) can read these to enrich their execution context.

| Env Var | Source | Description |
|---------|--------|-------------|
| `OPENCLAW_SCOPE_ID` | `scope.scope_id` | Routing scope identifier |
| `OPENCLAW_THREAD_ID` | `scope.thread_id` | Thread identifier |
| `OPENCLAW_SESSION_ID` | `scope.session_id` | Session identifier |
| `OPENCLAW_GIT_BRANCH` | `context.git_branch` | Git branch name |
| `OPENCLAW_WORKING_DIR` | `context.working_directory` | Working directory |
| `OPENCLAW_PROTOCOL_VERSION` | `protocol_version` | Bridge protocol version |
| `OPENCLAW_ATTACHMENTS` | `attachments[].path` | Colon-separated absolute file paths |
| `OPENCLAW_MODEL_PREFS` | `context.model_preferences` | JSON string of model preferences |
| `OPENCLAW_LANGUAGE` | `context.language` | Primary programming language |
| `OPENCLAW_TASK_TYPE` | `context.task_type` | Task type hint |
| `OPENCLAW_CONTINUITY` | `continuity` | JSON string of continuity metadata |

## Stability Guarantees

- **New optional fields** may be added to input/output without a version bump.
- **Existing fields** will not change type or meaning within a major protocol version.
- **Env vars** prefixed with `OPENCLAW_` are reserved. The router may add new ones at any time.
- **Breaking changes** (field removal, type changes) require a `protocol_version` bump.

## See Also

- `router/models.py` — TaskMeta, ExecutorResult dataclass definitions
- `bin/ai-code-runner` — entrypoint implementation
