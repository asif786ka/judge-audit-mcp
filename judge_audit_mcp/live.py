"""Optional live judge adapter.

Off by default, and off hard: `JUDGE_AUDIT_LIVE=1` plus a provider API key. Every
other module in this package is deterministic, offline, and free, and that is the
property worth protecting — an audit tool whose own results move between runs is
in no position to lecture anyone about reproducibility.

What this buys you: closing the loop. Instead of emit → run your judge → feed
back, you point it at a judge config and it does all three. Useful for a live
demo, and for the case where you don't have a judge harness handy at all.

What it costs, stated plainly because the default matters:

  - **Money.** A position probe over 200 items is 400 judge calls, twice, because
    both orders need judging.
  - **Nondeterminism.** Even at temperature 0, providers do not guarantee
    identical outputs across time. Findings from live runs are not reproducible
    in the way the offline path's are.
  - **Untestable in CI.** No API key in the test runner, so this path is smoke-
    tested with a stub rather than the real thing.

The offline path is the product. This is a convenience.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from .models import JudgeFingerprint, JudgeRecord

DEFAULT_JUDGE_PROMPT = (
    "You are grading a response for quality.\n\n"
    "Question:\n{question}\n\nResponse:\n{response}\n\n"
    "Rubric:\n{rubric}\n\n"
    "Reply with a single integer score from {scale_min} to {scale_max} and nothing else."
)


class LiveDisabled(Exception):
    """Raised when live judging is requested but not enabled."""


def is_enabled() -> bool:
    return os.environ.get("JUDGE_AUDIT_LIVE", "").strip() in ("1", "true", "yes", "on")


def _require_enabled() -> None:
    if not is_enabled():
        raise LiveDisabled(
            "Live judging is off. This server's default path is offline: it reads "
            "judge outputs you already have and costs nothing. To let it call a "
            "judge itself, set JUDGE_AUDIT_LIVE=1 and provide ANTHROPIC_API_KEY or "
            "OPENAI_API_KEY. Note that live findings are not reproducible run to run.")


# --- provider adapters -------------------------------------------------------

def _anthropic_caller(model: str, temperature: float) -> Callable[[str], str]:
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise LiveDisabled(
            "The `anthropic` package isn't installed. It is deliberately not a "
            "dependency of this server — the offline path needs no SDK. "
            "`pip install anthropic` to use the live adapter.") from e
    import anthropic
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        r = client.messages.create(
            model=model, max_tokens=16, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return call


def _openai_caller(model: str, temperature: float) -> Callable[[str], str]:
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise LiveDisabled(
            "The `openai` package isn't installed. `pip install openai` to use the "
            "live adapter, or use the offline path, which needs no SDK.") from e
    import openai
    client = openai.OpenAI()

    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, max_tokens=16, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""
    return call


def get_caller(provider: str, model: str, temperature: float = 0.0) -> Callable[[str], str]:
    _require_enabled()
    p = (provider or "").lower()
    if p in ("anthropic", "claude", ""):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LiveDisabled("ANTHROPIC_API_KEY is not set.")
        return _anthropic_caller(model, temperature)
    if p in ("openai", "gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise LiveDisabled("OPENAI_API_KEY is not set.")
        return _openai_caller(model, temperature)
    raise LiveDisabled(f"Unknown provider {provider!r}. Supported: anthropic, openai.")


# --- scoring loop ------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(text: str, lo: float, hi: float) -> Optional[float]:
    """Pull a score out of a judge reply.

    Tolerant of the judge ignoring "reply with only a number", which it will.
    Tries JSON first, then the first in-range number. Returns None rather than
    guessing — an unparseable reply is missing data, and silently coercing it to
    a number would fabricate exactly the kind of quiet error this server exists
    to catch.
    """
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for k in ("score", "rating", "grade", "value"):
                if k in obj:
                    return float(obj[k])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    for m in _NUM.finditer(t):
        v = float(m.group())
        if lo - 1e-9 <= v <= hi + 1e-9:
            return v
    return None


def judge_items(
    items: list[dict],
    model: str,
    provider: str = "anthropic",
    rubric: str = "Score the response for correctness, clarity and usefulness.",
    prompt_template: str = DEFAULT_JUDGE_PROMPT,
    scale_min: float = 1,
    scale_max: float = 10,
    temperature: float = 0.0,
    hint_field: str = "_hint",
) -> tuple[list[JudgeRecord], list[str]]:
    """Score `items` live. Each item needs item_id and output_text; question optional.

    Returns records stamped with the fingerprint of the judge that produced them —
    which is the point. A live run that didn't record its own fingerprint would be
    this server committing the exact sin it audits.
    """
    call = get_caller(provider, model, temperature)
    fp = JudgeFingerprint.from_config(
        model=model, prompt=prompt_template, rubric=rubric,
        scale=f"{scale_min:g}-{scale_max:g}", temperature=temperature, provider=provider,
    )

    out: list[JudgeRecord] = []
    errors: list[str] = []
    for it in items:
        prompt = prompt_template.format(
            question=it.get("question", ""), response=it.get("output_text", ""),
            rubric=rubric, scale_min=f"{scale_min:g}", scale_max=f"{scale_max:g}",
        )
        hint = it.get(hint_field) or ""
        if hint:
            prompt = f"{hint}\n\n{prompt}"
        try:
            raw = call(prompt)
        except Exception as e:                      # noqa: BLE001 — surfaced, not swallowed
            errors.append(f"{it.get('item_id')}: {type(e).__name__}: {e}")
            continue
        sc = parse_score(raw, scale_min, scale_max)
        if sc is None:
            errors.append(f"{it.get('item_id')}: unparseable judge reply {raw[:60]!r}")
            continue
        out.append(JudgeRecord(
            item_id=str(it.get("item_id")),
            score=sc,
            output_family=it.get("output_family", ""),
            output_text=it.get("output_text", ""),
            variant=it.get("variant", ""),
            fingerprint=fp,
        ))
    return out, errors
