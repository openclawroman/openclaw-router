# Operational Runbooks

This document provides standalone guides for installing, configuring, monitoring, and testing the OpenClaw Router.

---

## 1. Installation

### Prerequisites

- Python 3.10+
- `pip` (or `pip3`)
- Git
- A POSIX-compatible shell (bash, zsh)

### Clone the Repository

```bash
git clone https://github.com/openclawroman/openclaw-router.git
cd openclaw-router
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
# If no requirements.txt, install pytest for testing:
pip3 install pytest
```

### Smoke Test

Run the full test suite to verify everything works:

```bash
python3 -m pytest tests/ -q
```

Expected output: all tests pass (e.g. `400+ passed`). If any test fails, check that your Python version meets the prerequisites and that you're in the repository root.

**Verification:** Confirm the exit code is 0:

```bash
python3 -m pytest tests/ -q && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED"
```

---

## 2. Manual State Override

The router uses a 4-state subscription-aware system. You can manually override the state or let the router manage it automatically. (Previously the system used `normal` and `last10` states — these have been superseded by the 4-state architecture.)

### How to Enter Conservation Mode

```bash
# Manual switch via state file
echo '{"state": "openai_conservation"}' > config/codex_manual_state.json
```

### How to Switch to Claude Backup

```bash
echo '{"state": "claude_backup"}' > config/codex_manual_state.json
```

### How to Switch to OpenRouter Fallback

```bash
echo '{"state": "openrouter_fallback"}' > config/codex_manual_state.json
```

### How to Return to Primary

```bash
echo '{"state": "openai_primary"}' > config/codex_manual_state.json
```

### Clear Manual Override

Remove the manual override to return to automatic state resolution:

```bash
rm -f config/codex_manual_state.json
```

### Verify Current State

```bash
# Check auto state
cat config/codex_auto_state.json | python3 -m json.tool

# Check manual override (if set)
cat config/codex_manual_state.json | python3 -m json.tool

# Programmatic check
python3 -c "
from router.policy import resolve_state
print(f'Current state: {resolve_state().value}')
"
```

**Valid states:** `openai_primary`, `openai_conservation`, `claude_backup`, `openrouter_fallback`

---

## 3. Automatic State Transitions

### What Happens When OpenAI Rate Limits Hit

The router detects `rate_limited` errors from the OpenAI backend. Behavior depends on frequency:

1. **Single rate limit**: Tries next executor in the chain (fallback within current state)
2. **Repeated rate limits**: Can auto-transition from `openai_primary` → `openai_conservation`
3. **Persistent exhaustion**: Escalates from `openai_conservation` → `claude_backup`

The router uses **sticky state** — it won't bounce back to `openai_primary` after one success. A sustained cooldown period or manual override is required.

### What Happens When Claude Is Unavailable

The router tracks Claude Code health in `runtime/claude_health.json`. When Claude degrades:

1. If in `claude_backup` state: auto-transitions to `openrouter_fallback`
2. If in `openai_primary` or `openai_conservation`: Claude is skipped in the chain, OpenRouter handles fallback
3. Recovery: When Claude health improves, the router can restore Claude in chains (but won't auto-escalate back to `claude_backup` state)

### Budget Pressure Signals

The router monitors these signals for automatic transitions:

- `rate_limited` — Provider returned 429
- `quota_exhausted` — Provider returned 402 or quota exceeded
- `auth_error` — Authentication failure
- Task count per provider (rolling window)
- Time-since-last-success

**Note:** Manual override always takes precedence over automatic transitions.

---

## 3. Provider Setup

### Codex CLI (OpenAI Native)

1. Install the Codex CLI:
   ```bash
   npm install -g @openai/codex
   ```

2. Authenticate:
   ```bash
   codex auth login
   ```

3. Verify:
   ```bash
   codex --version
   ```

**Verification:** `codex --version` should return a version number without errors.

### Claude Code (Anthropic)

1. Install Claude Code:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. Set the API key:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-your-key-here"
   # Persist in your shell profile:
   echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.zshrc
   ```

3. Verify:
   ```bash
   claude --version
   ```

**Verification:** `claude --version` returns a version. Test a quick invocation:
```bash
claude -p "say hello" --max-tokens 10
```

### OpenRouter

1. Create an account at [openrouter.ai](https://openrouter.ai).

2. Generate an API key from the dashboard.

3. Fund your account (credits required for model calls).

4. Set the key:
   ```bash
   export OPENROUTER_API_KEY="sk-or-your-key-here"
   echo 'export OPENROUTER_API_KEY="sk-or-your-key-here"' >> ~/.zshrc
   ```

5. Verify:
   ```bash
   curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']), 'models available')"
   ```

**Verification:** The curl command should list the number of available models without errors.

---

## 4. Monitoring

### Tail the Routing Log

Watch routing decisions in real-time:

```bash
tail -f runtime/routing.jsonl | python3 -m json.tool --no-ensure-ascii
```

Or for a compact view (one line per entry):

```bash
tail -f runtime/routing.jsonl
```

### Check Claude Health

The router writes `runtime/claude_health.json` after each Claude invocation:

```bash
cat runtime/claude_health.json | python3 -m json.tool
```

### Error Rate Queries

Count errors by type from the routing log:

```bash
# Total routes
wc -l < runtime/routing.jsonl

# Failed routes
python3 -c "
import json
from collections import Counter
errors = Counter()
with open('runtime/routing.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        result = entry.get('result', {})
        if not result.get('success', True):
            err = result.get('normalized_error', 'unknown')
            errors[err] += 1
for err, count in errors.most_common():
    print(f'  {err}: {count}')
"

# Success rate
python3 -c "
import json
total = success = 0
with open('runtime/routing.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        result = entry.get('result', {})
        if result:
            total += 1
            if result.get('success'):
                success += 1
rate = (success / total * 100) if total else 0
print(f'{success}/{total} = {rate:.1f}% success rate')
"
```

### Route Quality Report

Generate a comprehensive report using the built-in telemetry module:

```bash
python3 -c "
from router.telemetry import RouteQualityReporter
r = RouteQualityReporter()
report = r.generate_report()
print(f'Total routes: {report.total_routes}')
print(f'Success rate: {report.overall_success_rate:.1%}')
print(f'Avg latency: {report.avg_latency_ms:.0f} ms')
print(f'Total cost: \${report.total_cost_usd:.4f}')
print(f'Top errors: {report.most_common_errors[:5]}')
"
```

**Verification:** The commands should produce output without errors. The quality report should show non-zero totals if routes have been logged.

---

## 5. End-to-End Dry Runs

### Dry Run 1: OpenAI Primary State Routing

Simulate a routing decision in openai_primary state:

```bash
python3 -c "
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality
from router.policy import build_chain, RouterState

task = TaskMeta(
    task_id='dry-001',
    agent='coder',
    task_class=TaskClass.IMPLEMENTATION,
    risk=TaskRisk.LOW,
    modality=TaskModality.TEXT,
    summary='add a hello world function',
)
chain = build_chain(task, RouterState.OPENAI_PRIMARY)
print('Chain (openai_primary):')
for i, entry in enumerate(chain):
    print(f'  {i+1}. {entry.tool}:{entry.backend} ({entry.model_profile})')
assert len(chain) == 3, 'Primary chain should have 3 entries'
assert chain[0].tool == 'codex_cli', 'First entry should be codex_cli'
assert chain[1].tool == 'claude_code', 'Second entry should be claude_code'
print('DRY RUN 1 PASSED')
"
```

**Verification:** Output shows 3-entry chain starting with `codex_cli:openai_native`, and ends with "DRY RUN 1 PASSED".

### Dry Run 2: Claude Backup State Routing

```bash
python3 -c "
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality
from router.policy import build_chain, RouterState

task = TaskMeta(
    task_id='dry-002',
    agent='coder',
    task_class=TaskClass.BUGFIX,
    risk=TaskRisk.MEDIUM,
    modality=TaskModality.TEXT,
    summary='fix null pointer error',
)
chain = build_chain(task, RouterState.CLAUDE_BACKUP)
print('Chain (claude_backup):')
for i, entry in enumerate(chain):
    print(f'  {i+1}. {entry.tool}:{entry.backend} ({entry.model_profile})')
assert len(chain) == 2, 'Claude backup chain should have 2 entries'
assert chain[0].tool == 'claude_code', 'Claude backup first entry should be claude_code'
print('DRY RUN 2 PASSED')
"
```

**Verification:** Output shows 2-entry chain starting with `claude_code:anthropic`, and ends with "DRY RUN 2 PASSED".

### Dry Run 3: OpenRouter Fallback State Routing

```bash
python3 -c "
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality
from router.policy import build_chain, RouterState

task = TaskMeta(
    task_id='dry-003',
    agent='coder',
    task_class=TaskClass.IMPLEMENTATION,
    risk=TaskRisk.LOW,
    modality=TaskModality.TEXT,
    summary='simple task',
)
chain = build_chain(task, RouterState.OPENROUTER_FALLBACK)
print('Chain (openrouter_fallback):')
for i, entry in enumerate(chain):
    print(f'  {i+1}. {entry.tool}:{entry.backend} ({entry.model_profile})')
assert len(chain) == 1, 'OpenRouter fallback chain should have 1 entry'
assert chain[0].backend == 'openrouter', 'Should route through openrouter'
print('DRY RUN 3 PASSED')
"
```

**Verification:** Output shows 1-entry chain through OpenRouter, and ends with "DRY RUN 3 PASSED".

### Dry Run 4: Logging Round-Trip

Test that the logger writes and the reporter reads correctly:

```bash
python3 -c "
from pathlib import Path
from router.logger import RoutingLogger
from router.telemetry import RouteQualityReporter
from router.models import TaskMeta, TaskClass, ChainEntry, RouteDecision, ExecutorResult

log_path = Path('/tmp/dry-run-routing.jsonl')
log_path.unlink(missing_ok=True)

logger = RoutingLogger(log_path=log_path)
task = TaskMeta(task_id='dry-004', agent='coder', task_class=TaskClass.IMPLEMENTATION, summary='dry run test')
decision = RouteDecision(
    task_id='dry-004', state='openai_primary',
    chain=[ChainEntry(tool='codex_cli', backend='openai_native', model_profile='codex_primary')],
    reason='dry run', attempted_fallback=False,
)
result = ExecutorResult(
    task_id='dry-004', tool='codex_cli', backend='openai_native',
    model_profile='codex_primary', success=True, latency_ms=500,
    cost_estimate_usd=0.01,
)
logger.log(task, decision, result=result)

reporter = RouteQualityReporter(log_path=log_path)
report = reporter.generate_report()
assert report.total_routes == 1
assert report.success_count == 1
assert report.overall_success_rate == 1.0
assert report.total_cost_usd == 0.01
print('DRY RUN 4 PASSED')

log_path.unlink(missing_ok=True)
"
```

**Verification:** Script prints "DRY RUN 4 PASSED" with no errors.

### Dry Run 5: Fallback Eligibility

```bash
python3 -c "
from router.policy import can_fallback, ELIGIBLE_FALLBACK_ERRORS

eligible = ['auth_error', 'rate_limited', 'quota_exhausted', 'provider_unavailable', 'provider_timeout', 'transient_network_error']
ineligible = ['toolchain_error', 'unknown', None]

for err in eligible:
    assert can_fallback(err), f'{err} should be eligible'
for err in ineligible:
    assert not can_fallback(err), f'{err} should NOT be eligible'

print(f'Tested {len(eligible)} eligible + {len(ineligible)} ineligible error types')
print('DRY RUN 5 PASSED')
"
```

**Verification:** Script prints "DRY RUN 5 PASSED" confirming fallback logic works correctly.

---

## 6. Config Validation

The router includes a config validation system that checks `config/router.config.json` at startup and can be run manually via CLI.

### CLI Flag: `--validate-config`

```bash
./bin/ai-code-runner --validate-config
```

### What Validation Checks

The validator performs the following checks:

- **Required top-level keys**: `version`, `models`, `state`, `tools`
- **Version compatibility**: `version` must be in `SUPPORTED_VERSIONS` (currently `{1, 2}`)
- **Required model sections**: `openrouter`, `codex`, `claude` must be present under `models`
- **State configuration**: `state.default`, `state.manual_state_file`, `state.auto_state_file` must exist
- **Default state validity**: `state.default` must be one of `openai_primary`, `openai_conservation`, `claude_backup`, `openrouter_fallback`
- **Required tools**: `codex_cli` must be present; `claude_code` is a warning if missing
- **Reliability config types**: `chain_timeout_s`, `max_fallbacks`, and circuit breaker params must be numbers
- **Unknown keys**: Warns on any unrecognized top-level keys

### How to Read Error Messages

Each error/warning line follows the format:

```
  [severity] path: message
```

- **`[error]`** — Config is invalid and the router will refuse to start
- **`[warning]`** — Config is valid but has a potential issue (e.g. empty model section)

The `path` tells you exactly where in the config the problem is (e.g. `models.openrouter`, `state.default`, `reliability.circuit_breaker.threshold`).

### Example Validation Output

**Valid config:**
```
Config OK
```

**Invalid config:**
```
Config INVALID: 2 error(s), 1 warning(s)
  [error] version: Unsupported version 99. Supported: {1, 2}
  [error] state.default: Invalid default state 'bad_state'. Valid: {'openai_primary', 'openai_conservation', 'claude_backup', 'openrouter_fallback'}
  [warning] unknown_section: Unknown top-level key: 'unknown_section'
```

### Programmatic Usage

```python
from router.config_validator import validate_config_file

result = validate_config_file("config/router.config.json")
if not result.valid:
    for err in result.errors:
        print(f"[{err.severity}] {err.path}: {err.message}")
```

### Note on VALID_STATES

The `VALID_STATES` set in `config_validator.py` is kept in sync with the `RouterState` enum in `router/policy.py`. If you add a new state to the router, remember to update both places. (Future improvement: import `VALID_STATES` directly from the model.)
