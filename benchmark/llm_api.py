"""Small wrapper around the Anthropic Messages API used by the benchmark
(question generation and the LLM judge).

* The API key comes from ``.env`` (``ANTHROPIC_API_KEY`` or ``CLAUDE_API_KEY``,
  read with ``dotenv_values``) or the environment; it is never printed.
* No sampling parameters are sent (the current models reject them).
* Only ``text`` blocks carry the answer (``thinking`` blocks are skipped) and
  ``stop_reason`` is checked (``refusal`` / ``max_tokens``) before parsing.
* Transient errors (429, 5xx/overloaded, timeouts, connection errors) are
  retried with exponential backoff; authentication / permission / not-found
  errors are fatal (``LLMFatalError``).
* JSON is parsed robustly (code fences, surrounding prose).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# USD per million tokens (input, output); used for cost estimates only.
PRICING = {"claude-sonnet-5-5": (2.00, 10.00)}


class LLMError(RuntimeError):
    """A single request failed (refusal, truncation, unparsable output, bad request...)."""

    def __init__(self, message: str, status: str = "error", info: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.info = info or {}


class LLMFatalError(LLMError):
    """The API cannot be used at all (missing/invalid key, no permission, unknown model)."""


def _is_placeholder(value: str) -> bool:
    lowered = (value or "").strip().lower()
    return (not lowered) or lowered.startswith(("your_", "your-", "<")) or "placeholder" in lowered \
        or lowered in ("changeme", "none", "null", "dummy")


def get_api_key(env_file: Optional[Path] = None) -> Optional[str]:
    """ANTHROPIC_API_KEY / CLAUDE_API_KEY from the environment, else from ``.env`` (read with
    ``dotenv_values``). As everywhere in the project, existing environment variables win."""
    values: Dict[str, Optional[str]] = {}
    path = Path(env_file) if env_file else ROOT / ".env"
    if path.exists():
        try:
            from dotenv import dotenv_values

            values = dict(dotenv_values(path))
        except Exception:  # dotenv missing/broken: fall back to the environment
            values = {}
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        for value in (os.environ.get(var), values.get(var)):
            if value and not _is_placeholder(value):
                return value.strip()
    return None


def make_client(api_key: Optional[str] = None, env_file: Optional[Path] = None):
    """An ``anthropic.Anthropic`` client, or None when no usable key is configured."""
    key = api_key or get_api_key(env_file)
    if not key:
        return None
    import anthropic

    # the SDK's own retries are disabled: call_text() retries with typed handling
    return anthropic.Anthropic(api_key=key, max_retries=0, timeout=300.0)


def check_access(client, model: str) -> Tuple[bool, str]:
    """Cheap credential/model check (no tokens billed). Returns (ok, reason)."""
    import anthropic

    try:
        client.models.retrieve(model)
        return True, "ok"
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        return False, f"authentication failed ({type(exc).__name__}, HTTP {getattr(exc, 'status_code', '?')})"
    except anthropic.NotFoundError:
        return False, f"model {model!r} not found for this key"
    except anthropic.APIConnectionError as exc:
        return False, f"connection error ({type(exc).__name__})"
    except anthropic.APIStatusError as exc:
        return False, f"API error HTTP {exc.status_code}"


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_loose(text: str) -> Any:
    """Parse a JSON object from model output: plain JSON, fenced JSON or JSON inside prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty output")
    candidates = [text]
    candidates += [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
    # first balanced {...} (or [...]) block, string-aware
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for pos in range(start, len(text)):
                ch = text[pos]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:pos + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    raise ValueError("no JSON object found in model output")


def _usage_dict(usage: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, key, None) if usage is not None else None
        if isinstance(value, int):
            out[key] = value
    return out


def estimate_cost(model: str, usage: Dict[str, int]) -> float:
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    tokens_in = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) \
        + usage.get("cache_read_input_tokens", 0)
    return tokens_in / 1e6 * price_in + usage.get("output_tokens", 0) / 1e6 * price_out


def _sleep_for(attempt: int, exc: Exception, base: float = 2.0, cap: float = 60.0) -> float:
    retry_after = None
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            retry_after = float(response.headers.get("retry-after"))
        except (TypeError, ValueError, AttributeError):
            retry_after = None
    delay = retry_after if retry_after is not None else min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, 0.5)


def call_text(client, *, model: str, system: str, user: str, max_tokens: int = 16000,
              output_config: Optional[Dict[str, Any]] = None, max_attempts: int = 5,
              sleep=time.sleep) -> Tuple[str, Dict[str, Any]]:
    """One non-streaming request; returns (text, info). Raises LLMError / LLMFatalError."""
    import anthropic

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if output_config:
        # SDK 0.57 predates output_config: send it as an extra body field
        kwargs["extra_body"] = {"output_config": output_config}
    last: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(**kwargs)
            break
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMFatalError(f"authentication failed ({type(exc).__name__})", status="auth") from exc
        except anthropic.NotFoundError as exc:
            raise LLMFatalError(f"model {model!r} not found", status="not_found") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"bad request: {str(getattr(exc, 'message', exc))[:300]}", status="bad_request") from exc
        except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as exc:
            last = exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in (408, 409, 429, 529):
                last = exc
            else:
                raise LLMError(f"API error HTTP {exc.status_code}", status="api_error") from exc
        if attempt + 1 < max_attempts:
            sleep(_sleep_for(attempt, last))
    else:
        raise LLMError(f"giving up after {max_attempts} attempts: {type(last).__name__}", status="retries_exhausted")

    info: Dict[str, Any] = {
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": _usage_dict(getattr(response, "usage", None)),
        "model": getattr(response, "model", model),
        "request_id": getattr(response, "_request_id", None),
    }
    if info["stop_reason"] == "refusal":
        details = getattr(response, "stop_details", None)
        info["refusal_category"] = getattr(details, "category", None) if details is not None else None
        raise LLMError("the model declined the request", status="refusal", info=info)
    if info["stop_reason"] == "max_tokens":
        raise LLMError("output truncated (max_tokens)", status="max_tokens", info=info)
    text = "".join(getattr(block, "text", "") or "" for block in getattr(response, "content", []) or []
                   if getattr(block, "type", None) == "text")
    return text, info


def call_json(client, *, model: str, system: str, user: str, max_tokens: int = 16000,
              effort: Optional[str] = None, schema: Optional[Dict[str, Any]] = None,
              sleep=time.sleep) -> Tuple[Any, Dict[str, Any]]:
    """Request JSON output. With ``schema`` structured outputs are used; if the API rejects
    the format it falls back to a plain request (the prompt must ask for JSON only)."""
    config: Dict[str, Any] = {}
    if effort:
        config["effort"] = effort
    if schema is not None:
        config["format"] = {"type": "json_schema", "schema": schema}
    try:
        text, info = call_text(client, model=model, system=system, user=user, max_tokens=max_tokens,
                               output_config=config or None, sleep=sleep)
        info["structured"] = schema is not None
    except LLMError as exc:
        if exc.status != "bad_request" or schema is None:
            raise
        fallback = {"effort": effort} if effort else None
        text, info = call_text(client, model=model, system=system, user=user, max_tokens=max_tokens,
                               output_config=fallback, sleep=sleep)
        info["structured"] = False
        info["structured_error"] = str(exc)[:200]
    try:
        return parse_json_loose(text), info
    except ValueError as exc:
        raise LLMError(f"unparsable JSON output: {exc}", status="parse_error", info=info) from exc
