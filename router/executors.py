"""Executor adapters for Codex CLI, Claude Code, and OpenRouter."""

import subprocess
import os
from typing import Optional

from .models import TaskMeta, RouteDecision, Executor
from .errors import (
    CodexQuotaError, CodexAuthError, CodexToolError,
    ClaudeQuotaError, ClaudeAuthError, ClaudeToolError,
    OpenRouterError
)


# Environment: use same env as current process
ENV = os.environ.copy()


def run_codex(meta: TaskMeta) -> RouteDecision:
    """
    Run Codex CLI.
    
    Returns normalized result:
    - ok: success
    - quota_error: hit plan limits
    - auth_error: login issue
    - tool_error: execution failure
    """
    try:
        # Codex CLI usage - adjust based on actual CLI interface
        # This is a placeholder - actual implementation depends on Codex CLI specifics
        result = subprocess.run(
            ["codex", meta.task_brief],
            cwd=meta.repo_path,
            env=ENV,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return RouteDecision(
                executor=Executor.CODEX_CLI,
                model="codex-default",
                reason="codex: execution successful"
            )
        elif "quota" in result.stderr.lower() or "limit" in result.stderr.lower():
            raise CodexQuotaError()
        elif "auth" in result.stderr.lower() or "unauthorized" in result.stderr.lower():
            raise CodexAuthError()
        else:
            raise CodexToolError(result.stderr or "Unknown error")
            
    except FileNotFoundError:
        # Codex CLI not installed
        raise CodexToolError("Codex CLI not found in PATH")
    except subprocess.TimeoutExpired:
        raise CodexToolError("Codex execution timed out")


def run_claude(meta: TaskMeta) -> RouteDecision:
    """
    Run Claude Code.
    
    Returns normalized result.
    """
    try:
        # Claude Code CLI - adjust based on actual CLI interface
        result = subprocess.run(
            ["claude", "-p", meta.task_brief],
            cwd=meta.repo_path,
            env=ENV,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return RouteDecision(
                executor=Executor.CLAUDE_CODE,
                model="claude-default",
                reason="claude: execution successful"
            )
        elif "quota" in result.stderr.lower() or "limit" in result.stderr.lower():
            raise ClaudeQuotaError()
        elif "auth" in result.stderr.lower() or "unauthorized" in result.stderr.lower():
            raise ClaudeAuthError()
        else:
            raise ClaudeToolError(result.stderr or "Unknown error")
            
    except FileNotFoundError:
        raise ClaudeToolError("Claude Code CLI not found in PATH")
    except subprocess.TimeoutExpired:
        raise ClaudeToolError("Claude execution timed out")


def run_openrouter(meta: TaskMeta, model: str = "moonshotai/kimi-k2.5") -> RouteDecision:
    """
    Run OpenRouter API call using urllib (no external dependencies).
    
    model: either "moonshotai/kimi-k2.5" (large context) or "minimax/minimax-m2.5" (cheap)
    """
    import urllib.request
    import urllib.error
    import json
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY not set")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": meta.task_brief}],
        "max_tokens": 4000
    }).encode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openclaw.local",
        "X-Title": "OpenClaw Router"
    }
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status == 200:
                return RouteDecision(
                    executor=Executor.OPENROUTER,
                    model=model,
                    reason=f"openrouter: execution successful ({model})"
                )
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise OpenRouterError("Rate limited")
        elif e.code == 401:
            raise OpenRouterError("Invalid API key")
        else:
            raise OpenRouterError(f"API error: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        raise OpenRouterError(f"Request failed: {e.reason}")