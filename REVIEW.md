# Code Review: 4-State Subscription Architecture (PR #18)

## Summary

The 4-state subscription-aware architecture is **well-designed and functionally correct**. All 616 tests pass, backward compatibility is preserved, and the new states make logical sense for budget management. However, there are a few issues worth addressing before merge — one silent-failure risk in `get_model()`, duplicated definitions between modules, a type annotation bug, and unused config routing sections.

**Overall verdict: Approve with minor fixes.**

## ✅ Correct

### State Machine
- All 4 state chains are built correctly with the expected fallback order:
  - `openai_primary`: codex → claude → openrouter ✓
  - `openai_conservation`: codex → claude → openrouter ✓
  - `claude_backup`: claude → openrouter ✓
  - `openrouter_fallback`: openrouter only ✓

### Model Selection
- `choose_openai_profile()` returns `gpt-5.4` for critical/debug/arch, `gpt-5.4-mini` otherwise ✓
- `choose_openai_profile()` in conservation mode returns `gpt-5.4-mini` by default, `gpt-5.4` only for critical ✓
- `choose_claude_model()` returns Sonnet default, Opus for critical/arch ✓
- `choose_openrouter_profile()` correctly selects Kimi for multimodal, MiMo for critical/debug/arch, MiniMax default ✓
- `run_claude()` correctly passes `--model` flag to CLI ✓
- `run_codex()` correctly passes `--model` flag to CLI ✓

### Backward Compatibility
- `CodexState.NORMAL` aliases `OPENAI_PRIMARY` ✓
- `CodexState.LAST10` aliases `CLAUDE_BACKUP` ✓
- `_build_normal_chain` and `_build_last10_chain` exist and delegate correctly ✓
- `resolve_state()` accepts optional `store` parameter ✓
- `_validate_state("normal")` returns `OPENAI_PRIMARY` ✓
- `_validate_state("last10")` returns `CLAUDE_BACKUP` ✓
- Config contains `"normal"` and `"last10"` routing sections as documentation ✓

### Config
- All 4 new routing sections present in config ✓
- MiMo model string: `"xiaomi/mimo-v2-pro"` ✓
- Sonnet: `"claude-sonnet-4.6"`, Opus: `"claude-opus-4.6"` ✓
- Default state is `"openai_primary"` ✓

### Test Coverage
- 47 new tests covering all 4 states, model selection, backward compat, state resolution ✓
- Tests use `tmp_path` properly for isolation ✓
- All 616 tests pass (616 total, 0 failures) ✓

## ⚠️ Issues Found

### 1. `get_model()` silently returns unknown profile names (MEDIUM)

In `router/config_loader.py:52`:
```python
def get_model(profile: str) -> str:
    # ...
    # Fallback: return profile name as-is
    return profile
```

If a profile name is mistyped (e.g. `get_model("sonnnet")`), it silently returns `"sonnnet"` as a model string. This will pass all tests that check the config path but fail at runtime when the CLI or API receives an invalid model name.

**Recommendation:** Add a `KeyError` raise or at minimum a `logging.warning()` when the fallback is hit. This would catch typos during development and testing.

### 2. Duplicated `ELIGIBLE_FALLBACK_ERRORS` and `can_fallback()` (LOW)

`ELIGIBLE_FALLBACK_ERRORS` and `can_fallback()` are defined in both `router/errors.py` and `router/policy.py`. The policy module imports from its own local definitions rather than reusing the ones from `errors.py`. This creates a maintenance risk: if someone adds a new eligible error type to `errors.py`, they must remember to also update `policy.py`.

**Recommendation:** Have `policy.py` import `ELIGIBLE_FALLBACK_ERRORS` and `can_fallback` from `errors.py` instead of duplicating them.

### 3. `run_openrouter()` type annotation bug (LOW)

In `router/executors.py:327`:
```python
def run_openrouter(
    meta: TaskMeta,
    model: str = None,  # ← should be Optional[str]
    profile: str = "openrouter_minimax",
) -> ExecutorResult:
```

The type hint says `str` but the default is `None`. Should be `model: Optional[str] = None`.

### 4. Config routing sections are unused by `build_chain()` (LOW — Informational)

The 4 routing sections in `config/router.config.json` (e.g. `"openai_primary"`, `"openai_conservation"`) define static chains, but `build_chain()` in `policy.py` builds chains **programmatically** based on task metadata — it never reads the config routing sections.

The config chains and the programmatic chains can diverge. For example, the config `openai_primary` chain uses `"model_profile": "auto"` for codex, but the code produces `"codex_gpt54"` or `"codex_gpt54_mini"` depending on task.

**Recommendation:** Either (a) add a comment in the config noting these are documentation/reference only, or (b) remove the config routing sections if they serve no purpose. Leaving them as-is invites confusion.

### 5. `StateError` not imported in `state_store.py` (LOW)

The file does `from .errors import StateError` which works — verified. Not actually an issue, but noting for completeness that the error hierarchy is clean.

## 🧪 Test Coverage

**Coverage: Good (47 tests for new functionality)**

| Area | Covered? | Notes |
|------|----------|-------|
| 4-state enum values | ✅ | `TestCodexStateEnum` |
| Backward compat aliases | ✅ | `NORMAL == OPENAI_PRIMARY`, `LAST10 == CLAUDE_BACKUP` |
| StateStore validation | ✅ | All 4 states + backward compat strings |
| Chain building (all 4 states) | ✅ | Tool order, entry count, model profiles |
| OpenAI model selection | ✅ | Via chain model_profile checks |
| Claude model selection | ✅ | `TestClaudeModelSelection` — Sonnet/Opus |
| OpenRouter profile selection | ✅ | `TestOpenRouterProfileSelection` — MiniMax/Kimi/MiMo |
| State resolution priority | ✅ | Manual > Auto > Default |
| Config model strings | ✅ | `TestNewModelConfig` |

**Missing test coverage:**
- No test for `get_model()` fallback behavior (silent profile name return)
- No test for `run_claude()` with model parameter (integration-level)
- No test for `_run_executor()` dispatch with the new model profiles
- No test for `choose_openai_profile()` behavior in conservation mode (only chain-level tests)
- Edge case: `_validate_state("")` or `_validate_state(None)` — `None` would cause `TypeError` since it's passed to `CodexState(mapped)` after lookup

## 📋 Recommended Fixes

### Before Merge (Required)
None — the code is functional and correct.

### Before Next Release (Recommended)
1. **`get_model()` silent fallback** — Add a warning or exception when the profile is not found in any model section
2. **Deduplicate `can_fallback()`** — Import from `errors.py` in `policy.py`
3. **Fix `run_openrouter()` type hint** — `model: Optional[str] = None`
4. **Document config routing sections** — Add a comment clarifying they're reference only, not used by `build_chain()`

### Future Consideration
- Add an integration test that exercises `_run_executor()` with each new model profile
- Consider validating that `get_model()` calls succeed during router startup (fail-fast)
