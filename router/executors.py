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
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from .models import TaskMeta, ExecutorResult
from .errors import (
    CodexQuotaError, CodexAuthError, CodexToolError,
    ClaudeQuotaError, ClaudeAuthError, ClaudeToolError,
    OpenRouterError, ProviderTimeoutError, normalize_error
)
from .secrets import sanitize_secrets


ENV = os.environ.copy()


def _save_output_files(task_id: str, stdout: str, stderr: str) -> tuple[list[str], Optional[str], Optional[str]]:
    """Save stdout/stderr to /tmp/{task_id}.*.txt files.

    Returns (artifacts, stdout_ref, stderr_ref).
    """
    artifacts: list[str] = []
    stdout_ref: Optional[str] = None
    stderr_ref: Optional[str] = None

    if task_id:
        if stdout:
            path = f"/tmp/{task_id}.stdout.txt"
            Path(path).write_text(stdout, encoding="utf-8")
            artifacts.append(path)
            stdout_ref = path
        if stderr:
            path = f"/tmp/{task_id}.stderr.txt"
            Path(path).write_text(stderr, encoding="utf-8")
            artifacts.append(path)
            stderr_ref = path

    return artifacts, stdout_ref, stderr_ref


def _update_claude_health(success: bool, error_type: Optional[str] = None) -> None:
    """Write runtime/claude_health.json after each Claude invocation."""
    runtime_dir = Path("runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if success:
        health = {
            "available": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": "recent successful invocation",
        }
    else:
        health = {
            "available": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": error_type or "unknown error",
        }

    (runtime_dir / "claude_health.json").write_text(
        json.dumps(health, indent=2), encoding="utf-8"
    )

from .config_loader import get_model


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


def _run_subprocess(cmd: list, cwd: str, timeout: int = 300) -> tuple[int, str, str, bool]:
    """Run a subprocess, return (returncode, stdout, stderr, timed_out)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=ENV,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr, False
    except subprocess.TimeoutExpired:
        return -1, "", "Execution timed out", True
    except FileNotFoundError as e:
        return -1, "", str(e), False


# ---------------------------------------------------------------------------
# Codex CLI - OpenAI native
# ---------------------------------------------------------------------------

def run_codex(meta: TaskMeta, model: Optional[str] = None) -> ExecutorResult:
    """Run Codex CLI via OpenAI native backend.

    Args:
        meta: Task metadata.
        model: Optional model string override (e.g. "gpt-5.4" or "gpt-5.4-mini").
               If None, falls back to the default codex model.
    """
    task_id = meta.task_id or ""
    cwd = meta.cwd or meta.repo_path

    cmd = ["codex"]
    if model:
        cmd.extend(["--model", model])
    cmd.append(meta.summary or task_id)

    start = time.time()
    returncode, stdout, stderr, timed_out = _run_subprocess(
        cmd,
        cwd=cwd
    )
    latency_ms = int((time.time() - start) * 1000)

    artifacts, stdout_ref, stderr_ref = _save_output_files(task_id, stdout, stderr)

    if returncode == 0:
        return _make_result(
            task_id=task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            exit_code=0,
            latency_ms=latency_ms,
            cost_estimate_usd=0.0,
            artifacts=artifacts,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            final_summary=stdout[:500] if stdout else None,
        )
    elif timed_out:
        raise ProviderTimeoutError("Codex execution timed out")
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
        model=get_model("minimax"),
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
        model=get_model("kimi"),
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
            cost = resp_data.get("usage", {}).get("cost") or resp_data.get("cost") or 0.0
            save_content = json.dumps({
                "request": payload_dict,
                "response": resp_data,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "cost_estimate_usd": cost,
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
                cost_estimate_usd=cost,
                artifacts=[filepath],
                stdout_ref=filepath,
            )
    except urllib.error.HTTPError as e:
        latency_ms = int((time.time() - start) * 1000)
        body = sanitize_secrets(e.read().decode("utf-8", errors="replace"))
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
        err_msg = f"Request failed: {sanitize_secrets(str(e.reason))}"
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

def run_claude(meta: TaskMeta, *, model: Optional[str] = None) -> ExecutorResult:
    """Run Claude Code (Anthropic backend).

    Args:
        meta: Task metadata.
        model: Optional model string override (e.g. "claude-sonnet-4.6" or "claude-opus-4.6").
    """
    task_id = meta.task_id or ""
    cwd = meta.cwd or meta.repo_path

    cmd = ["claude", "-p", meta.summary or task_id]
    if model:
        cmd.extend(["--model", model])

    start = time.time()
    returncode, stdout, stderr, timed_out = _run_subprocess(
        cmd,
        cwd=cwd
    )
    latency_ms = int((time.time() - start) * 1000)

    artifacts, stdout_ref, stderr_ref = _save_output_files(task_id, stdout, stderr)

    if returncode == 0:
        _update_claude_health(success=True)
        return _make_result(
            task_id=task_id,
            tool="claude_code",
            backend="anthropic",
            model_profile="claude_primary",
            success=True,
            exit_code=0,
            latency_ms=latency_ms,
            cost_estimate_usd=0.0,
            artifacts=artifacts,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            final_summary=stdout[:500] if stdout else None,
        )
    elif timed_out:
        _update_claude_health(success=False, error_type="provider_timeout")
        raise ProviderTimeoutError("Claude execution timed out")
    elif "quota" in stderr.lower() or "limit" in stderr.lower():
        _update_claude_health(success=False, error_type="quota_exhausted")
        raise ClaudeQuotaError()
    elif "auth" in stderr.lower() or "unauthorized" in stderr.lower():
        _update_claude_health(success=False, error_type="auth_error")
        raise ClaudeAuthError()
    else:
        _update_claude_health(success=False, error_type="toolchain_error")
        raise ClaudeToolError(stderr or "Unknown error")


# ---------------------------------------------------------------------------
# OpenRouter (generic)
# ---------------------------------------------------------------------------

def run_openrouter(
    meta: TaskMeta,
    model: Optional[str] = None,
    profile: str = "openrouter_minimax",
) -> ExecutorResult:
    """Run OpenRouter API call with specified model."""
    if model is None:
        model = get_model("minimax")
    return _openrouter_request(
        meta=meta,
        model=model,
        tool="openrouter",
        backend="openrouter",
        model_profile=profile,
    )
