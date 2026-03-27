"""Tests for architecture spec: file layout, doc structure, module imports, and config validity."""

import json
import sys
import unittest
from pathlib import Path

# Add parent to path so we can import router modules
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestFileLayout(unittest.TestCase):
    """Verify all expected files exist in the repo."""

    EXPECTED_FILES = [
        "router/models.py",
        "router/classifier.py",
        "router/policy.py",
        "router/executors.py",
        "router/errors.py",
        "router/state_store.py",
        "router/logger.py",
        "bin/ai-code-runner",
        "config/router.config.json",
    ]

    def test_all_expected_files_exist(self):
        missing = []
        for rel_path in self.EXPECTED_FILES:
            full_path = REPO_ROOT / rel_path
            if not full_path.exists():
                missing.append(rel_path)
        self.assertEqual(missing, [], f"Missing expected files: {missing}")

    def test_router_package_init(self):
        self.assertTrue((REPO_ROOT / "router" / "__init__.py").exists())


class TestArchitectureDocExists(unittest.TestCase):
    """Verify docs/architecture.md exists and has content."""

    def test_doc_exists(self):
        doc_path = REPO_ROOT / "docs" / "architecture.md"
        self.assertTrue(doc_path.exists(), "docs/architecture.md does not exist")

    def test_doc_has_content(self):
        doc_path = REPO_ROOT / "docs" / "architecture.md"
        content = doc_path.read_text()
        self.assertGreater(len(content), 500, "docs/architecture.md seems too short")

    def test_doc_not_empty(self):
        doc_path = REPO_ROOT / "docs" / "architecture.md"
        content = doc_path.read_text().strip()
        self.assertTrue(len(content) > 0, "docs/architecture.md is empty")


class TestArchitectureDocSections(unittest.TestCase):
    """Verify docs/architecture.md contains key sections."""

    @classmethod
    def setUpClass(cls):
        doc_path = REPO_ROOT / "docs" / "architecture.md"
        cls.content = doc_path.read_text()

    def test_has_plane1_section(self):
        self.assertIn("Plane 1", self.content, "Missing Plane 1 (OpenClaw Orchestration) section")

    def test_has_plane2_section(self):
        self.assertIn("Plane 2", self.content, "Missing Plane 2 (ai-code-runner) section")

    def test_has_state_machine_section(self):
        self.assertIn("State Machine", self.content, "Missing State Machine section")

    def test_has_normal_state(self):
        self.assertIn("normal", self.content, "Missing normal state description")

    def test_has_last10_state(self):
        self.assertIn("last10", self.content, "Missing last10 state description")

    def test_has_routing_contract(self):
        self.assertIn("Routing Contract", self.content, "Missing Routing Contract section")

    def test_has_error_handling(self):
        self.assertIn("Error", self.content, "Missing error handling section")

    def test_has_file_layout(self):
        self.assertIn("File Layout", self.content, "Missing File Layout section")

    def test_has_tool_backends(self):
        self.assertIn("Backend", self.content, "Missing tool backends section")

    def test_mentions_codex_cli(self):
        self.assertIn("codex_cli", self.content, "Missing codex_cli references")

    def test_mentions_claude_code(self):
        self.assertIn("claude_code", self.content, "Missing claude_code references")

    def test_mentions_openrouter(self):
        self.assertIn("openrouter", self.content.lower(), "Missing OpenRouter references")

    def test_has_health_tracking(self):
        self.assertIn("Health", self.content, "Missing health tracking section")

    def test_has_logging_section(self):
        self.assertIn("Logging", self.content, "Missing logging section")


class TestModuleImports(unittest.TestCase):
    """Verify all modules import successfully."""

    def test_import_models(self):
        import router.models
        self.assertTrue(hasattr(router.models, "TaskMeta"))
        self.assertTrue(hasattr(router.models, "RouteDecision"))
        self.assertTrue(hasattr(router.models, "ExecutorResult"))

    def test_import_classifier(self):
        import router.classifier
        self.assertTrue(hasattr(router.classifier, "Classifier") or
                        hasattr(router.classifier, "classify"))

    def test_import_policy(self):
        import router.policy
        self.assertTrue(hasattr(router.policy, "route_task") or
                        hasattr(router.policy, "build_chain"))

    def test_import_executors(self):
        import router.executors
        self.assertTrue(hasattr(router.executors, "run_codex") or
                        hasattr(router.executors, "run_claude"))

    def test_import_errors(self):
        import router.errors
        self.assertTrue(hasattr(router.errors, "RouterError") or
                        hasattr(router.errors, "normalize_error"))

    def test_import_state_store(self):
        import router.state_store
        self.assertTrue(hasattr(router.state_store, "StateStore"))

    def test_import_logger(self):
        import router.logger
        self.assertTrue(hasattr(router.logger, "RoutingLogger"))


class TestConfigValid(unittest.TestCase):
    """Verify config/router.config.json is valid JSON with required keys."""

    @classmethod
    def setUpClass(cls):
        config_path = REPO_ROOT / "config" / "router.config.json"
        with open(config_path) as f:
            cls.config = json.load(f)

    def test_is_valid_json(self):
        self.assertIsInstance(self.config, dict)

    def test_has_version(self):
        self.assertIn("version", self.config)

    def test_has_tools(self):
        self.assertIn("tools", self.config)

    def test_has_retry(self):
        self.assertIn("retry", self.config)

    def test_has_logging(self):
        self.assertIn("logging", self.config)

    def test_has_state(self):
        self.assertIn("state", self.config)


if __name__ == "__main__":
    unittest.main()
