"""Executor adapters for Codex CLI, Claude Code, and OpenRouter.

New spec internal tool roles:
  1. Codex CLI, OpenAI-native  -> tool=codex_cli,  backend=openai_native,  profile=codex_primary
  2. Claude Code, Anthropic    -> tool=claude_code, backend=anthropic,      profile=claude_primary
  3. Codex CLI, OpenRouter     -> tool=codex_cli,  backend=openrouter,      profile=openrouter_minimax
  4. Codex CLI, OpenRouter     -> tool=codex_cli,  backend=openrouter,      profile=openrouter_kimi
"""

import subprocess
import os
import uuid
import time
import urllib.request
import urllib.error
import json
from typing import Optional

from .models import (
    TaskMeta, RouteDecision, Executor, ExecutorBackend, ModelProfile
)
from .errors import (
    CodexQuotaError, CodexAuthError, CodexToolError,
    ClaudeQuotaError, ClaudeAuthError, ClaudeToolError,
    OpenRouterError, normalize_error
)


ENV = os.environ.copy()

# OpenRouter models
OPENROUTER_MINIMAX_MODEL = "minimax/minimax-m2.7"
OPENROUTER_KIMI_MODEL    = "moonshotai/kimi-k2.5"


def _make_decision(
    executor: Executor,
    backend: ExecutorBackend,
    model_profile: ModelProfile,
    model_str: str,
    reason: str,
    exit_code: int = 0,
    request_id: Optional[str] = None,
    cost_estimate_usd: Optional[float] = None,
    normalized_error: Optional[str] = None,
) -> RouteDecision:
    """Construct a RouteDecision with all the new fields populated."""
    return RouteDecision(
        executor=executor,
        backend=backend,
        model_profile=model_profile,
        model=model_str,
        chain=[],
        reason=reason,
        status="success" if normalized_error is None else "error",
        exit_code=exit_code,
        request_id=request_id,
        cost_estimate_usd=cost_estimate_usd,
        artifacts=[],
        normalized_error=normalized_error,
    )


def _run_subprocess(cmd: list, cwd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=ENV,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Execution timed out"
    except FileNotFoundError as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# Codex CLI - with profile support
# ---------------------------------------------------------------------------

def run_codex(meta: TaskMeta, model: str = "codex-default", profile: str = "openai_native") -> RouteDecision:
    """
    Run Codex CLI.
    
    Parameters
    ----------
    meta    : TaskMeta  – Task metadata.
    model   : str       – Model string (default "codex-default").
    profile : str       – "openai_native" (OAuth, no API key) or "openrouter".
    
    When profile is "openrouter", routes via OpenRouter using the Kimi model.
    """
    if profile == "openrouter":
        # Delegate to OpenRouter path
        return run_codex_openrouter_kimi(meta)

    # Default: openai_native profile — standard Codex CLI via OAuth
    cwd = meta.cwd or meta.repo_path
    returncode, stdout, stderr = _run_subprocess(
        ["codex", meta.task_brief],
        cwd=cwd
    )

    if returncode == 0:
        return _make_decision(
            Executor.CODEX_CLI,
            ExecutorBackend.OPENAI_NATIVE,
            ModelProfile.CODEX_PRIMARY,
            model,
            "codex: execution successful",
            exit_code=0,
        )
    elif "quota" in stderr.lower() or "limit" in stderr.lower():
        raise CodexQuotaError()
    elif "auth" in stderr.lower() or "unauthorized" in stderr.lower():
        raise CodexAuthError()
    else:
        raise CodexToolError(stderr or "Unknown error")


# ---------------------------------------------------------------------------
# Codex CLI via OpenRouter - MiniMax
# ---------------------------------------------------------------------------

def run_codex_openrouter_minimax(meta: TaskMeta) -> RouteDecision:
    """
    Codex CLI via OpenRouter using MiniMax model.
    profile=openrouter_minimax
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return _make_decision(
            Executor.CODEX_CLI,
            ExecutorBackend.OPENROUTER,
            ModelProfile.OPENROUTER_MINIMAX,
            OPENROUTER_MINIMAX_MODEL,
            "codex openrouter minimax: API key missing",
            exit_code=-1,
            normalized_error="auth_error",
        )

    return _openrouter_request(
        meta=meta,
        model=OPENROUTER_MINIMAX_MODEL,
        executor=Executor.CODEX_CLI,
        backend=ExecutorBackend.OPENROUTER,
        model_profile=ModelProfile.OPENROUTER_MINIMAX,
    )


# ---------------------------------------------------------------------------
# Codex CLI via OpenRouter - Kimi
# ---------------------------------------------------------------------------

def run_codex_openrouter_kimi(meta: TaskMeta) -> RouteDecision:
    """
    Codex CLI via OpenRouter using Kimi model.
    profile=openrouter_kimi
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return _make_decision(
            Executor.CODEX_CLI,
            ExecutorBackend.OPENROUTER,
            ModelProfile.OPENROUTER_KIMI,
            OPENROUTER_KIMI_MODEL,
            "codex openrouter kimi: API key missing",
            exit_code=-1,
            normalized_error="auth_error",
        )

    return _openrouter_request(
        meta=meta,
        model=OPENROUTER_KIMI_MODEL,
        executor=Executor.CODEX_CLI,
        backend=ExecutorBackend.OPENROUTER,
        model_profile=ModelProfile.OPENROUTER_KIMI,
    )


def _openrouter_request(
    meta: TaskMeta,
    model: str,
    executor: Executor,
    backend: ExecutorBackend,
    model_profile: ModelProfile,
) -> RouteDecision:
    """Make an OpenRouter API call and return RouteDecision."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    request_id = str(uuid.uuid4())
    
    payload = json.dumps({
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
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status == 200:
                return _make_decision(
                    executor,
                    backend,
                    model_profile,
                    model,
                    f"openrouter: execution successful ({model})",
                    exit_code=0,
                    request_id=request_id,
                )
            else:
                err_msg = f"HTTP {response.status}"
                return _make_decision(
                    executor,
                    backend,
                    model_profile,
                    model,
                    f"openrouter failed: {err_msg}",
                    exit_code=response.status,
                    normalized_error=normalize_error(err_msg),
                )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        err_msg = f"HTTP {e.code}: {body}"
        norm_err = normalize_error(err_msg)
        if e.code == 429:
            norm_err = "rate_limited"
        elif e.code == 401:
            norm_err = "auth_error"
        return _make_decision(
            executor,
            backend,
            model_profile,
            model,
            f"openrouter error: {err_msg}",
            exit_code=e.code,
            normalized_error=norm_err,
        )
    except urllib.error.URLError as e:
        err_msg = f"Request failed: {e.reason}"
        return _make_decision(
            executor,
            backend,
            model_profile,
            model,
            err_msg,
            exit_code=-1,
            normalized_error=normalize_error(err_msg),
        )


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

def run_claude(meta: TaskMeta, model: str = "claude-default") -> RouteDecision:
    """Run Claude Code (Anthropic backend)."""
    cwd = meta.cwd or meta.repo_path
    returncode, stdout, stderr = _run_subprocess(
        ["claude", "-p", meta.task_brief],
        cwd=cwd
    )

    if returncode == 0:
        return _make_decision(
            Executor.CLAUDE_CODE,
            ExecutorBackend.ANTHROPIC,
            ModelProfile.CLAUDE_PRIMARY,
            model,
            "claude: execution successful",
            exit_code=0,
        )
    elif "quota" in stderr.lower() or "limit" in stderr.lower():
        raise ClaudeQuotaError()
    elif "auth" in stderr.lower() or "unauthorized" in stderr.lower():
        raise ClaudeAuthError()
    else:
        raise ClaudeToolError(stderr or "Unknown error")


# ---------------------------------------------------------------------------
# OpenRouter (generic)
# ---------------------------------------------------------------------------

def run_openrouter(
    meta: TaskMeta,
    model: str = "minimax/minimax-m2.7",
    profile: ModelProfile = ModelProfile.OPENROUTER_MINIMAX,
) -> RouteDecision:
    """Run OpenRouter API call with specified model."""
    return _openrouter_request(
        meta=meta,
        model=model,
        executor=Executor.OPENROUTER,
        backend=ExecutorBackend.OPENROUTER,
        model_profile=profile,
    )