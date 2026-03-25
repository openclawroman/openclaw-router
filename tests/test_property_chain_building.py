"""Property-based tests for chain building invariants.

Tests that build_chain() produces valid chains for every (task_class, state)
combination, with correct structure and field presence.

Properties tested:
  - Every (task_class, state) combination produces a valid chain
  - Chain entries have required fields (tool, backend, model_profile)
  - Chain length is reasonable (1-4 entries)
  - OpenRouter only appears in openrouter_fallback state (as primary)
  - No duplicate entries within a chain
  - First entry is always the primary executor for that state
"""

import itertools
import random

import pytest

from router.models import (
    CodexState, TaskClass, TaskRisk, TaskModality, TaskMeta, ChainEntry,
)
from router.policy import build_chain, validate_chain

ALL_STATES = list(CodexState)
ALL_TASK_CLASSES = list(TaskClass)
ALL_RISKS = list(TaskRisk)

VALID_TOOLS = {"codex_cli", "claude_code", "openrouter"}
VALID_BACKENDS = {"openai_native", "anthropic", "openrouter"}
VALID_PROFILES = {
    "codex_primary", "codex_gpt54", "codex_gpt54_mini",
    "claude_primary", "claude_sonnet", "claude_opus",
    "openrouter_minimax", "openrouter_kimi", "openrouter_mimo", "openrouter_dynamic",
}

# State → expected first backend
STATE_FIRST_BACKEND = {
    CodexState.OPENAI_PRIMARY: "openai_native",
    CodexState.OPENAI_CONSERVATION: "openai_native",
    CodexState.CLAUDE_BACKUP: "anthropic",
    CodexState.OPENROUTER_FALLBACK: "openrouter",
}


def _make_task(task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, **kwargs):
    return TaskMeta(
        task_id="chain-test",
        task_class=task_class,
        risk=risk,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Every (task_class, state) combination produces a valid chain
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllCombinationsValid:
    """build_chain works for every (TaskClass, CodexState) pair."""

    @pytest.mark.parametrize("task_class,state", list(itertools.product(ALL_TASK_CLASSES, ALL_STATES)))
    def test_combination_produces_chain(self, task_class, state):
        """Each (task_class, state) pair yields a non-empty chain."""
        task = _make_task(task_class=task_class)
        chain = build_chain(task, state)
        assert isinstance(chain, list)
        assert len(chain) >= 1

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_validate_chain_passes(self, state):
        """Chain passes validate_chain() for each state."""
        task = _make_task()
        chain = build_chain(task, state)
        valid, reason = validate_chain(state, chain)
        assert valid is True, f"Chain invalid for {state}: {reason}"

    @pytest.mark.parametrize("task_class,state", list(itertools.product(ALL_TASK_CLASSES, ALL_STATES)))
    def test_validate_all_combinations(self, task_class, state):
        """Every (task_class, state) chain passes validate_chain()."""
        task = _make_task(task_class=task_class)
        chain = build_chain(task, state)
        valid, reason = validate_chain(state, chain)
        assert valid is True, f"Invalid chain for ({task_class}, {state}): {reason}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Chain entries have required fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainEntryFields:
    """Every chain entry has valid tool, backend, and model_profile."""

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_entries_have_required_fields(self, state):
        """All entries have non-empty tool, backend, model_profile."""
        task = _make_task()
        chain = build_chain(task, state)
        for i, entry in enumerate(chain):
            assert isinstance(entry, ChainEntry)
            assert entry.tool, f"Entry {i} has empty tool"
            assert entry.backend, f"Entry {i} has empty backend"
            assert entry.model_profile, f"Entry {i} has empty model_profile"

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_tool_is_valid(self, state):
        """All entry tools are in the known set."""
        task = _make_task()
        chain = build_chain(task, state)
        for entry in chain:
            assert entry.tool in VALID_TOOLS, f"Unknown tool: {entry.tool}"

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_backend_is_valid(self, state):
        """All entry backends are in the known set."""
        task = _make_task()
        chain = build_chain(task, state)
        for entry in chain:
            assert entry.backend in VALID_BACKENDS, f"Unknown backend: {entry.backend}"

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_profile_is_valid(self, state):
        """All entry model_profiles are in the known set."""
        task = _make_task()
        chain = build_chain(task, state)
        for entry in chain:
            assert entry.model_profile in VALID_PROFILES, f"Unknown profile: {entry.model_profile}"

    def test_random_combinations_have_valid_fields(self):
        """Stress: 500 random (task, state) — all entries have valid fields."""
        for _ in range(500):
            task = _make_task(
                task_class=random.choice(ALL_TASK_CLASSES),
                risk=random.choice(ALL_RISKS),
                has_screenshots=random.random() < 0.1,
                requires_multimodal=random.random() < 0.1,
            )
            state = random.choice(ALL_STATES)
            chain = build_chain(task, state)
            for entry in chain:
                assert entry.tool in VALID_TOOLS
                assert entry.backend in VALID_BACKENDS
                assert entry.model_profile in VALID_PROFILES


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Chain length is reasonable (1-4 entries)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainLength:
    """Chain length is between 1 and 4 entries."""

    @pytest.mark.parametrize("task_class,state", list(itertools.product(ALL_TASK_CLASSES, ALL_STATES)))
    def test_length_in_bounds(self, task_class, state):
        """Chain has 1-4 entries."""
        task = _make_task(task_class=task_class)
        chain = build_chain(task, state)
        assert 1 <= len(chain) <= 4, f"Chain length {len(chain)} out of bounds for ({task_class}, {state})"

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_expected_lengths(self, state):
        """Each state has the expected chain length."""
        task = _make_task()
        chain = build_chain(task, state)
        if state == CodexState.OPENROUTER_FALLBACK:
            assert len(chain) == 1
        elif state == CodexState.CLAUDE_BACKUP:
            assert len(chain) == 2
        else:
            assert len(chain) == 3

    def test_stress_chain_lengths(self):
        """Stress: 500 random chains — all within bounds."""
        for _ in range(500):
            task = _make_task(
                task_class=random.choice(ALL_TASK_CLASSES),
                risk=random.choice(ALL_RISKS),
            )
            state = random.choice(ALL_STATES)
            chain = build_chain(task, state)
            assert 1 <= len(chain) <= 4


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OpenRouter only appears in openrouter_fallback state (as primary)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenRouterOnlyInFallback:
    """OpenRouter is the only backend in openrouter_fallback chains."""

    def test_openrouter_fallback_only_has_openrouter(self):
        """In openrouter_fallback, every entry uses openrouter backend."""
        task = _make_task()
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        for entry in chain:
            assert entry.backend == "openrouter", (
                f"Non-openrouter backend '{entry.backend}' in openrouter_fallback chain"
            )

    def test_openrouter_fallback_chain_is_single(self):
        """openrouter_fallback produces exactly 1 entry."""
        task = _make_task()
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert len(chain) == 1

    def test_openrouter_fallback_all_risks(self):
        """openrouter_fallback only contains openrouter for all risk levels."""
        for risk in ALL_RISKS:
            task = _make_task(risk=risk)
            chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
            for entry in chain:
                assert entry.backend == "openrouter"

    def test_claude_backup_has_no_openai_native(self):
        """claude_backup chain never contains openai_native backend."""
        task = _make_task()
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        for entry in chain:
            assert entry.backend != "openai_native", (
                f"openai_native in claude_backup chain: {entry.tool}:{entry.backend}"
            )

    def test_claude_backup_starts_with_anthropic(self):
        """claude_backup chain starts with anthropic backend."""
        task = _make_task()
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].backend == "anthropic"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. No duplicate entries within a chain
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoDuplicates:
    """No exact duplicate (tool, backend, profile) entries in a chain."""

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_no_duplicate_entries(self, state):
        """Chain has no duplicate (tool, backend, model_profile) tuples."""
        task = _make_task()
        chain = build_chain(task, state)
        seen = set()
        for entry in chain:
            key = (entry.tool, entry.backend, entry.model_profile)
            assert key not in seen, f"Duplicate chain entry: {key}"
            seen.add(key)

    def test_stress_no_duplicates(self):
        """Stress: 500 random chains — no duplicates."""
        for _ in range(500):
            task = _make_task(
                task_class=random.choice(ALL_TASK_CLASSES),
                risk=random.choice(ALL_RISKS),
            )
            state = random.choice(ALL_STATES)
            chain = build_chain(task, state)
            seen = set()
            for entry in chain:
                key = (entry.tool, entry.backend, entry.model_profile)
                assert key not in seen
                seen.add(key)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. First entry matches expected executor for each state
# ═══════════════════════════════════════════════════════════════════════════════

class TestFirstEntryExecutor:
    """First chain entry is the expected primary for each state."""

    @pytest.mark.parametrize("state,expected_backend", [
        (CodexState.OPENAI_PRIMARY, "openai_native"),
        (CodexState.OPENAI_CONSERVATION, "openai_native"),
        (CodexState.CLAUDE_BACKUP, "anthropic"),
        (CodexState.OPENROUTER_FALLBACK, "openrouter"),
    ])
    def test_first_entry_backend(self, state, expected_backend):
        """First entry uses the expected backend for this state."""
        task = _make_task()
        chain = build_chain(task, state)
        assert chain[0].backend == expected_backend

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_openrouter_fallback_first_is_openrouter(self, state):
        """First entry backend matches the state's primary provider."""
        task = _make_task()
        chain = build_chain(task, state)
        expected = STATE_FIRST_BACKEND[state]
        assert chain[0].backend == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Chain consistency across different task parameters
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainConsistency:
    """Chain structure is consistent regardless of task details (except profiles)."""

    def test_chain_length_consistent_by_state(self):
        """Chain length depends only on state, not task class or risk."""
        lengths = {}
        for state in ALL_STATES:
            task = _make_task()
            chain = build_chain(task, state)
            lengths[state] = len(chain)

        # Verify consistency across different tasks
        for state in ALL_STATES:
            for task_class in ALL_TASK_CLASSES:
                for risk in ALL_RISKS:
                    task = _make_task(task_class=task_class, risk=risk)
                    chain = build_chain(task, state)
                    assert len(chain) == lengths[state], (
                        f"Chain length changed for ({task_class}, {state}): "
                        f"expected {lengths[state]}, got {len(chain)}"
                    )

    def test_backend_order_consistent_by_state(self):
        """Backend sequence depends only on state."""
        for state in ALL_STATES:
            task1 = _make_task(risk=TaskRisk.LOW)
            task2 = _make_task(risk=TaskRisk.CRITICAL)
            chain1 = build_chain(task1, state)
            chain2 = build_chain(task2, state)

            backends1 = [e.backend for e in chain1]
            backends2 = [e.backend for e in chain2]
            assert backends1 == backends2, (
                f"Backend order differs for {state}: {backends1} vs {backends2}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Task multimodality affects OpenRouter profile selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultimodalProfileSelection:
    """Multimodal/screenshot tasks get appropriate OpenRouter profiles."""

    def test_screenshot_task_gets_kimi(self):
        """Screenshot tasks use openrouter_kimi profile."""
        task = _make_task(has_screenshots=True)
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert any(e.model_profile == "openrouter_kimi" for e in openrouter_entries)

    def test_critical_task_gets_mimo(self):
        """Critical risk tasks use openrouter_mimo profile."""
        task = _make_task(risk=TaskRisk.CRITICAL)
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert any(e.model_profile == "openrouter_mimo" for e in openrouter_entries)

    def test_normal_task_gets_minimax(self):
        """Normal tasks use openrouter_minimax profile."""
        task = _make_task(risk=TaskRisk.LOW)
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert any(e.model_profile == "openrouter_minimax" for e in openrouter_entries)
