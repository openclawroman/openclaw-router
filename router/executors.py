"""Executor adapters for Codex CLI, Claude Code, and OpenRouter.

Spec internal tool roles:
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

from .models import TaskMeta, ExecutorResult
from .errors import (
    CodexQuotaError, CodexAuthError, CodexToolError,
    ClaudeQuotaError, ClaudeAuthError, ClaudeToolError,
    OpenRouterError, normalize_error
)


ENV = os.environ.copy()

# OpenRouter models
OPENROUTER_MINIMAX_MODEL = "minimax/minimax-m2.7"
OPENROUTER_KIMI_MODEL    = "moonshotai/kimi-k2.5"


def _make_result(
    task_id: str,
    tool: str,
    backend: str,
    model_profile: str,
    success: bool = True,
    normalized_error: Optional[str] = None,
    exit_code: Optional[int] = None,
    latency_ms: int = 0,
    request_id: Optional[str] = None,
    cost_estimate_usd: Optional[float] = None,
    artifacts: Optional[list] = None,
    stdout_ref: Optional[str] = None,
    stderr_ref: Optional[str] = None,
    final_summary: Optional[str] = None,
) -> ExecutorResult:
    """Construct an ExecutorResult."""
    return ExecutorResult(
        task_id=task_id,
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=success,
        normalized_error=normalized_error,
        exit_code=exit_code,
        latency_ms=latency_ms,
        request_id=request_id,
        cost_estimate_usd=cost_estimate_usd,
        artifacts=artifacts or [],
        stdout_ref=stdout_ref,
        stderr_ref=stderr_ref,
        final_summary=final_summary,
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
# Codex CLI - OpenAI native
# ---------------------------------------------------------------------------

def run_codex(meta: TaskMeta) -> ExecutorResult:
    """Run Codex CLI via OpenAI native backend."""
    task_id = meta.task_id or ""
    cwd = meta.cwd or meta.repo_path

    start = time.time()
    returncode, stdout, stderr = _run_subprocess(
        ["codex", meta.summary or task_id],
        cwd=cwd
    )
    latency_ms = int((time.time() - start) * 1000)

    if returncode == 0:
        return _make_result(
            task_id=task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            exit_code=0,
            latency_ms=latency_ms,
            stdout_ref=stdout[:1000] if stdout else None,
            final_summary=stdout[:500] if stdout else None,
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

def run_codex_openrouter_minimax(meta: TaskMeta) -> ExecutorResult:
    """Codex CLI via OpenRouter using MiniMax model."""
    task_id = meta.task_id or ""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        filepath = _save_request_response(task_id, json.dumps({"error": "Missing OPENROUTER_API_KEY", "status": "auth_error"}, indent=2))
        return _make_result(
            task_id=task_id,
            tool="codex_cli",
            backend="openrouter",
            model_profile="openrouter_minimax",
            success=False,
            normalized_error="auth_error",
            exit_code=-1,
            artifacts=[filepath],
            stdout_ref=filepath,
        )

    return _openrouter_request(
        meta=meta,
        model=OPENROUTER_MINIMAX_MODEL,
        tool="codex_cli",
        backend="openrouter",
        model_profile="openrouter_minimax",
    )


# ---------------------------------------------------------------------------
# Codex CLI via OpenRouter - Kimi
# ---------------------------------------------------------------------------

def run_codex_openrouter_kimi(meta: TaskMeta) -> ExecutorResult:
    """Codex CLI via OpenRouter using Kimi model."""
    task_id = meta.task_id or ""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        filepath = _save_request_response(task_id, json.dumps({"error": "Missing OPENROUTER_API_KEY", "status": "auth_error"}, indent=2))
        return _make_result(
            task_id=task_id,
            tool="codex_cli",
            backend="openrouter",
            model_profile="openrouter_kimi",
            success=False,
            normalized_error="auth_error",
            exit_code=-1,
            artifacts=[filepath],
            stdout_ref=filepath,
        )

    return _openrouter_request(
        meta=meta,
        model=OPENROUTER_KIMI_MODEL,
        tool="codex_cli",
        backend="openrouter",
        model_profile="openrouter_kimi",
    )


def _save_request_response(task_id: str, content: str) -> str:
    """Save request/response content to /tmp/{task_id}.stdout.txt and return the path."""
    filepath = f"/tmp/{task_id}.stdout.txt"
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def _openrouter_request(
    meta: TaskMeta,
    model: str,
    tool: str,
    backend: str,
    model_profile: str,
) -> ExecutorResult:
    """Make an OpenRouter API call and return ExecutorResult."""
    task_id = meta.task_id or ""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    request_id = str(uuid.uuid4())

    payload_dict = {
        "model": model,
        "messages": [{"role": "user", "content": meta.summary or task_id}],
        "max_tokens": 4000
    }
    payload = json.dumps(payload_dict).encode('utf-8')

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openclaw.local",
        "X-Title": "OpenClaw Router"
    }

    start = time.time()
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            latency_ms = int((time.time() - start) * 1000)
            body = response.read().decode("utf-8", errors="replace")
            resp_data = json.loads(body) if body else {}
            save_content = json.dumps({
                "request": payload_dict,
                "response": resp_data,
                "latency_ms": latency_ms,
                "request_id": request_id,
            }, indent=2)
            filepath = _save_request_response(task_id, save_content)
            return _make_result(
                task_id=task_id,
                tool=tool,
                backend=backend,
                model_profile=model_profile,
                success=True,
                exit_code=0,
                latency_ms=latency_ms,
                request_id=request_id,
                artifacts=[filepath],
                stdout_ref=filepath,
            )
    except urllib.error.HTTPError as e:
        latency_ms = int((time.time() - start) * 1000)
        body = e.read().decode("utf-8", errors="replace")
        err_msg = f"HTTP {e.code}: {body}"
        norm_err = normalize_error(err_msg)
        if e.code == 429:
            norm_err = "rate_limited"
        elif e.code == 401:
            norm_err = "auth_error"
        elif e.code == 503:
            norm_err = "provider_unavailable"
        save_content = json.dumps({
            "request": payload_dict,
            "error": body,
            "status_code": e.code,
            "latency_ms": latency_ms,
            "request_id": request_id,
        }, indent=2)
        filepath = _save_request_response(task_id, save_content)
        return _make_result(
            task_id=task_id,
            tool=tool,
            backend=backend,
            model_profile=model_profile,
            success=False,
            normalized_error=norm_err,
            exit_code=e.code,
            latency_ms=latency_ms,
            request_id=request_id,
            artifacts=[filepath],
            stdout_ref=filepath,
        )
    except urllib.error.URLError as e:
        latency_ms = int((time.time() - start) * 1000)
        err_msg = f"Request failed: {e.reason}"
        save_content = json.dumps({
            "request": payload_dict,
            "error": str(e.reason),
            "latency_ms": latency_ms,
        }, indent=2)
        filepath = _save_request_response(task_id, save_content)
        return _make_result(
            task_id=task_id,
            tool=tool,
            backend=backend,
            model_profile=model_profile,
            success=False,
            normalized_error=normalize_error(err_msg),
            exit_code=-1,
            latency_ms=latency_ms,
            artifacts=[filepath],
            stdout_ref=filepath,
        )


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

def run_claude(meta: TaskMeta) -> ExecutorResult:
    """Run Claude Code (Anthropic backend)."""
    task_id = meta.task_id or ""
    cwd = meta.cwd or meta.repo_path

    start = time.time()
    returncode, stdout, stderr = _run_subprocess(
        ["claude", "-p", meta.summary or task_id],
        cwd=cwd
    )
    latency_ms = int((time.time() - start) * 1000)

    if returncode == 0:
        return _make_result(
            task_id=task_id,
            tool="claude_code",
            backend="anthropic",
            model_profile="claude_primary",
            success=True,
            exit_code=0,
            latency_ms=latency_ms,
            stdout_ref=stdout[:1000] if stdout else None,
            final_summary=stdout[:500] if stdout else None,
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
    profile: str = "openrouter_minimax",
) -> ExecutorResult:
    """Run OpenRouter API call with specified model."""
    return _openrouter_request(
        meta=meta,
        model=model,
        tool="openrouter",
        backend="openrouter",
        model_profile=profile,
    )
