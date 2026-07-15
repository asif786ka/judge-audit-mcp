"""Core data models for judge-audit-mcp.

The organising idea, inherited from its two siblings: a claim isn't just "true",
it is true *within a scope*, and the scope is the whole product.

    mobile-docs-mcp     a symbol is real   within a VERSION RANGE   VersionRange.status_at(version)
    store-preflight-mcp a rule   is in force within a DATE WINDOW   PolicyWindow.status_on(date)
    judge-audit-mcp     a score  is comparable within a FINGERPRINT JudgeFingerprint.comparable_to(other)

`JudgeFingerprint.comparable_to()` is the direct analogue of `status_at()` and
`status_on()`. Same model, third axis — because a judge score is not a
measurement of the thing being judged. It is a measurement of the thing being
judged *as read by a particular judge under a particular rubric*, and the moment
any part of that changes, the scale changes underneath you and the numbers stop
being commensurable.

That distinction is load-bearing, and it is the entire reason this server exists.
"My eval score went up 6%" has no truth value. It has a truth value *under a
fixed fingerprint*. Bump the judge model, retune one line of the rubric, nudge
temperature, and the 6% is a measurement of your rubric edit, not of your system.
Every eval platform on the market will report that 6% to you without a word of
warning, because the judge is infrastructure and nobody instruments the ruler.

The second design commitment, inherited from store-preflight's `confidence`
tiers: findings carry their own epistemics. Every number here is an *estimate
from a sample*, and a sample of 12 supports almost no claim at all. A tool that
reports "κ = 0.61, substantial agreement!" from 14 items is worse than no tool,
because it launders noise into a number with a Greek letter on it and someone
puts it in a slide. So findings carry `n`, a confidence interval, and a
confidence tier that *downgrades itself* when underpowered — the analogue of
`PolicyWindow.severity_on()` refusing to call something a blocker before its
deadline bites.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# --- severity ----------------------------------------------------------------
# What this finding means for whether you can trust the eval.

SEV_INVALIDATING = "invalidating"  # Your comparison is meaningless. Do not ship this number.
SEV_SERIOUS = "serious"            # Real bias, large enough to flip conclusions.
SEV_MINOR = "minor"                # Detectable, unlikely to change a decision alone.
SEV_CLEAN = "clean"                # Probed, nothing found.

_SEV_ORDER = {SEV_CLEAN: 0, SEV_MINOR: 1, SEV_SERIOUS: 2, SEV_INVALIDATING: 3}


def severity_at_least(sev: str, floor: str) -> bool:
    return _SEV_ORDER.get(sev, 0) >= _SEV_ORDER.get(floor, 0)


def worst_severity(sevs) -> str:
    out = SEV_CLEAN
    for s in sevs:
        if _SEV_ORDER.get(s, 0) > _SEV_ORDER.get(out, 0):
            out = s
    return out


# --- confidence --------------------------------------------------------------
# Research finding that shaped this: the bias literature (Zheng et al. 2023 on
# position bias; Wang et al. 2023 on order effects; Panickssery et al. 2024 on
# self-preference) reports effects that are real but *modest* — position-swap
# disagreement in the 10-30% range, self-preference deltas well under a point on
# a 10-point scale. Effects that size are trivially manufactured by sampling
# noise at n=20. So the tier is not decoration; it is the difference between this
# server being useful and it being a random-number generator with good branding.
#
# CERTAIN is reserved for claims that are *mechanically* true and need no
# statistics at all — "the judge model string differs between these two runs" is
# not an inference, it's a string comparison. That asymmetry matters: the single
# most damaging finding this tool makes (your fingerprint changed, your numbers
# are incomparable) is also the only one that needs no sample size to support.

CONF_CERTAIN = "certain"      # Mechanically true. A string compare, not an inference.
CONF_LIKELY = "likely"        # CI excludes the null, and n clears the power floor.
CONF_HEURISTIC = "heuristic"  # Suggestive. Underpowered or CI straddles null. Do not gate CI on it.

# Below this many paired observations, no statistical finding may claim better
# than CONF_HEURISTIC, whatever its p-value. Chosen, not inherited: at n=30 a
# paired comparison has ~80% power to detect a ~0.5σ effect, which is roughly the
# smallest effect that would actually change anyone's mind about a judge.
POWER_FLOOR_PAIRED = 30

# Cohen's κ is noisier than raw agreement and its variance blows up when one
# category dominates — precisely the regime real eval sets live in, where 80% of
# items pass. 50 is the conventional floor and still generous.
POWER_FLOOR_KAPPA = 50


def temper_confidence(conf: str, n: int, floor: int) -> str:
    """Downgrade a statistical claim that its sample size cannot support.

    The analogue of `PolicyWindow.severity_on()`: the claim isn't edited by hand
    as evidence accumulates, the model does it. A `likely` finding from 11 items
    is silently demoted to `heuristic`, so the CI gate can't fail a build on it.
    """
    if conf == CONF_CERTAIN:
        return conf  # Mechanical truths don't need a sample.
    if n < floor:
        return CONF_HEURISTIC
    return conf


# --- interpretation bands ----------------------------------------------------
# Landis & Koch (1977). Widely used, widely criticised as arbitrary — included
# because reviewers expect the vocabulary, labelled as a convention rather than
# a law because that's what it is.

def kappa_band(k: float) -> str:
    if k < 0.0:
        return "poor (worse than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


# --- the load-bearing model --------------------------------------------------

@dataclass(frozen=True)
class JudgeFingerprint:
    """Everything about a judge that, if changed, invalidates comparison.

    This is the third axis of the trilogy, and every field here earns its place
    by being something people change *without thinking of it as changing the
    measuring instrument*:

        model        : the obvious one. "We upgraded to the new Sonnet" is a
                       scale change, and it is never in the changelog of the
                       thing being evaluated.
        prompt_hash  : someone tightened one sentence of the judge instructions
                       in a PR titled "clarify judge prompt". The scale moved.
        rubric_hash  : the scoring criteria themselves.
        scale        : 1-5 vs 1-10 vs pass/fail. Rescaling is not linear in
                       judge behaviour, no matter what the arithmetic says.
        temperature  : a nonzero-temperature judge is a noisy instrument; two
                       runs of the *same* judge aren't even comparable to each
                       other without averaging.
        provider     : same weights behind a different API can serve different
                       quantisation, defaults, or system-prompt injection.

    `hash()` deliberately covers all of them. There is no "minor" field — the
    whole point is that people believe their change was minor and are wrong.
    """
    model: str = ""
    prompt_hash: str = ""
    rubric_hash: str = ""
    scale: str = ""              # e.g. "1-5" | "1-10" | "binary" | "pairwise"
    temperature: Optional[float] = None
    provider: str = ""
    version: str = ""            # optional pinned snapshot, e.g. "2025-06-01"

    @staticmethod
    def _h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else ""

    @classmethod
    def from_config(
        cls,
        model: str = "",
        prompt: str = "",
        rubric: str = "",
        scale: str = "",
        temperature: Optional[float] = None,
        provider: str = "",
        version: str = "",
    ) -> "JudgeFingerprint":
        """Build from raw text; prompt/rubric are hashed so we never store or
        leak the prompt itself. Whitespace-normalised, because a reflowed prompt
        is the same prompt and we'd rather not cry wolf on a formatter run."""
        return cls(
            model=model.strip(),
            prompt_hash=cls._h(" ".join(prompt.split())),
            rubric_hash=cls._h(" ".join(rubric.split())),
            scale=scale.strip(),
            temperature=temperature,
            provider=provider.strip(),
            version=version.strip(),
        )

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def is_empty(self) -> bool:
        """No fingerprint recorded at all — which is itself the most common and
        most damning finding. If you didn't record what judged your eval, you
        cannot ever prove your numbers were comparable, and no amount of
        statistics recovers that after the fact."""
        return not any((self.model, self.prompt_hash, self.rubric_hash,
                        self.scale, self.provider, self.version))

    def diff(self, other: "JudgeFingerprint") -> list[tuple[str, Any, Any]]:
        """Fields that differ. Mechanical, hence CONF_CERTAIN downstream."""
        out = []
        for f in ("model", "prompt_hash", "rubric_hash", "scale",
                  "temperature", "provider", "version"):
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                out.append((f, a, b))
        return out

    def comparable_to(self, other: "JudgeFingerprint") -> str:
        """One of: identical | unknown | incomparable.

        The direct analogue of `VersionRange.status_at()` / `PolicyWindow.status_on()`.

        Note what `unknown` means and what it does *not* mean. An unrecorded
        fingerprint is not evidence of sameness — it's absence of evidence, and
        the honest verdict is that you cannot know. Most eval harnesses land
        here, which is exactly why "our scores went up" is usually unfalsifiable
        rather than false.
        """
        if self.is_empty() or other.is_empty():
            return "unknown"
        if not self.diff(other):
            return "identical"
        return "incomparable"

    def describe(self) -> str:
        bits = []
        if self.model:
            bits.append(self.model)
        if self.provider:
            bits.append(f"via {self.provider}")
        if self.scale:
            bits.append(f"scale {self.scale}")
        if self.temperature is not None:
            bits.append(f"T={self.temperature}")
        if self.prompt_hash:
            bits.append(f"prompt:{self.prompt_hash[:8]}")
        if self.rubric_hash:
            bits.append(f"rubric:{self.rubric_hash[:8]}")
        if self.version:
            bits.append(self.version)
        return " · ".join(bits) if bits else "no fingerprint recorded"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "JudgeFingerprint":
        if not d:
            return JudgeFingerprint()
        t = d.get("temperature")
        return JudgeFingerprint(
            model=str(d.get("model", "") or ""),
            prompt_hash=str(d.get("prompt_hash", "") or ""),
            rubric_hash=str(d.get("rubric_hash", "") or ""),
            scale=str(d.get("scale", "") or ""),
            temperature=float(t) if t is not None and t != "" else None,
            provider=str(d.get("provider", "") or ""),
            version=str(d.get("version", "") or ""),
        )


# --- records -----------------------------------------------------------------

@dataclass
class JudgeRecord:
    """One judgement the judge emitted.

    Deliberately tolerant: real eval logs are a mess, and a tool that only reads
    perfectly-shaped data audits nobody's judge. `score` covers pointwise
    grading, `choice` covers pairwise A/B, and either may be absent.
    """
    item_id: str
    score: Optional[float] = None      # pointwise numeric grade
    choice: Optional[str] = None       # pairwise winner: "A" | "B" | "tie"
    label: Optional[str] = None        # categorical verdict: "pass"/"fail"/...
    output_family: str = ""            # model family that PRODUCED the output ("claude"/"gpt"/...)
    output_text: str = ""              # needed only by verbosity/length probes
    output_len: Optional[int] = None   # precomputed length; falls back to len(output_text)
    position: Optional[str] = None     # which slot this output occupied ("A"/"B")
    variant: str = ""                  # probe variant tag: ""|"swapped"|"padded"|"hinted"
    fingerprint: JudgeFingerprint = field(default_factory=JudgeFingerprint)
    meta: dict = field(default_factory=dict)

    def length(self) -> int:
        if self.output_len is not None:
            return int(self.output_len)
        return len(self.output_text or "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint.to_dict()
        return d


@dataclass
class GoldLabel:
    """A human judgement. The only thing in this system with any authority.

    `n_raters` and `rater_agreement` are here because gold isn't gold. If your
    humans only agree with *each other* at κ=0.5, then a judge scoring κ=0.45
    against them is roughly at the human noise ceiling and "the judge is bad" is
    the wrong conclusion — your task is underdefined. A calibration tool that
    ignores the ceiling will send people off to fix a judge that is already as
    good as a person, which is a genuinely expensive mistake.
    """
    item_id: str
    score: Optional[float] = None
    choice: Optional[str] = None
    label: Optional[str] = None
    n_raters: int = 1
    rater_agreement: Optional[float] = None   # inter-human κ, if you measured it
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- findings ----------------------------------------------------------------

@dataclass
class Finding:
    """One audit result. Never a bare number.

    Every field after `effect` exists to stop a number being quoted without its
    error bars. `ci_low`/`ci_high` and `n` travel with the estimate because the
    failure this whole server is about — quoting 6% as if it were a fact — is
    exactly what happens when they don't.
    """
    id: str
    title: str
    severity: str
    confidence: str
    detail: str = ""
    effect: Optional[float] = None       # the estimate (κ, mean delta, flip rate, ...)
    effect_label: str = ""               # what `effect` is, in words
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: int = 0
    p_value: Optional[float] = None
    evidence: list[str] = field(default_factory=list)
    fix: str = ""
    why: str = ""
    citation: str = ""
    tags: list[str] = field(default_factory=list)

    def ci_excludes(self, null: float = 0.0) -> Optional[bool]:
        """Does the interval exclude the null? None when we have no interval."""
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_low > null or self.ci_high < null

    def describe_effect(self) -> str:
        if self.effect is None:
            return ""
        s = f"{self.effect:+.3f}" if self.effect_label.startswith("delta") else f"{self.effect:.3f}"
        if self.ci_low is not None and self.ci_high is not None:
            s += f"  95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
        if self.n:
            s += f"  n={self.n}"
        return s

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditResult:
    """A set of findings plus the fingerprint context they were computed under."""
    kind: str                                   # "calibration" | "drift" | "bias"
    findings: list[Finding] = field(default_factory=list)
    fingerprint: JudgeFingerprint = field(default_factory=JudgeFingerprint)
    n_items: int = 0
    notes: list[str] = field(default_factory=list)

    def worst(self) -> str:
        return worst_severity(f.severity for f in self.findings)

    def counts(self) -> dict:
        c = {SEV_INVALIDATING: 0, SEV_SERIOUS: 0, SEV_MINOR: 0, SEV_CLEAN: 0}
        for f in self.findings:
            c[f.severity] = c.get(f.severity, 0) + 1
        return c

    def gating(self, floor: str) -> list[Finding]:
        """Findings that should fail a build.

        Heuristic findings never gate, regardless of severity. A tool that
        red-builds on an underpowered estimate gets turned off within a week,
        and then it audits nothing at all.
        """
        return [f for f in self.findings
                if severity_at_least(f.severity, floor) and f.confidence != CONF_HEURISTIC]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "fingerprint": self.fingerprint.to_dict(),
            "n_items": self.n_items,
            "counts": self.counts(),
            "worst": self.worst(),
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }
