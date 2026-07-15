"""judge-audit-mcp — who evaluates the evaluator?

Everyone runs LLM-as-judge. Almost nobody measures whether the judge agrees with
humans, whether it drifts when the judge model is bumped, or whether it exhibits
position, verbosity, self-preference or sycophancy bias.

Platforms ship judges. None of them ship judge QA.
"""
from .models import (
    AuditResult,
    Finding,
    GoldLabel,
    JudgeFingerprint,
    JudgeRecord,
)

__version__ = "0.1.1"

__all__ = ["AuditResult", "Finding", "GoldLabel", "JudgeFingerprint", "JudgeRecord",
           "__version__"]
