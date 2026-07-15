"""Score distribution pathology — can this judge even resolve the delta you're
reporting?

The only probe here that needs no variants, no human labels, and no extra judge
calls. It runs on any eval log you already have, right now. It is also the one
that invalidates the most results, which makes the ratio of its cost to its value
faintly absurd — the reason nobody runs it is that nobody thinks to look at the
judge's own output distribution at all.

What it looks for:

**Granularity collapse.** Your rubric says 1-10. Your judge emits 7, 8, and
occasionally 9. Its *effective* resolution — the perplexity of its score
distribution — is about 2, so you are running a 2-point judge with a 10-point
label on it. Every stakeholder reading "7.6 out of 10" believes they are looking
at a measurement with ten rungs. There are two.

**Quantisation.** A judge that emits only integers cannot produce a mean that
moves by 0.43 on a per-item basis. It produces a mean that moves because *some
whole number of items hopped a whole notch*. Your "+0.43 improvement" over 120
items is 52 items moving from 7 to 8 — and once it's phrased that way, the honest
next question is whether those 52 items are near a threshold the judge is
essentially coin-flipping on. Reporting the mean to two decimals disguises a
discrete, lumpy, threshold-driven process as a smooth one.

**Ceiling effects.** 60% of your items already score maximum. Real improvements
to those items cannot show up — there is nowhere up to go. Your eval is
structurally incapable of detecting the thing it's measuring, and it will report
"no significant change" forever while the system genuinely improves.

**Dead levels.** The judge has never once used levels 1-4. They are decoration.
The rubric text describing them has never influenced anything.

The unifying point: everyone audits the *model's* outputs and nobody audits the
*judge's* outputs, even though the judge's outputs are right there in the same
log, free.
"""
from __future__ import annotations

import math
from typing import Optional

from .. import stats as S
from ..io import load_judge_records
from ..models import (
    CONF_CERTAIN,
    AuditResult,
    Finding,
    SEV_CLEAN,
    SEV_INVALIDATING,
    SEV_MINOR,
    SEV_SERIOUS,
)

CEILING_SERIOUS = 0.40
CEILING_INVALIDATING = 0.60
# Effective levels below which the scale is a fiction. 3.0 is a judgement call:
# a judge behaving as a 3-point scale can still rank things, it just cannot
# support a claim quoted to two decimal places.
EFFECTIVE_SERIOUS = 3.0
EFFECTIVE_INVALIDATING = 2.0


def _is_integral(vals: list[float]) -> bool:
    return all(abs(v - round(v)) < 1e-9 for v in vals)


def probe_distribution(path: str, scale_min: Optional[float] = None,
                       scale_max: Optional[float] = None,
                       claimed_delta: Optional[float] = None, **kw) -> AuditResult:
    recs, rep = load_judge_records(path)
    res = AuditResult(kind="bias")
    res.notes.extend(rep.warnings)

    scores = [r.score for r in recs if r.score is not None]
    if not scores:
        res.notes.append("No numeric scores in this run.")
        return res

    if recs:
        res.fingerprint = recs[0].fingerprint
    res.n_items = len(scores)
    n = len(scores)

    lo = min(scores) if scale_min is None else scale_min
    hi = max(scores) if scale_max is None else scale_max
    # Prefer the declared rubric scale; a judge that never uses the bottom half is
    # exactly what we're hunting, so inferring the scale from its output would
    # define the pathology out of existence.
    if scale_min is None and scale_max is None and res.fingerprint.scale:
        try:
            a, b = res.fingerprint.scale.replace("–", "-").split("-")
            lo, hi = float(a), float(b)
        except (ValueError, AttributeError):
            pass

    eff = S.effective_levels(scores)
    used = sorted(set(scores))
    ceil = S.ceiling_share(scores, hi)
    floor_share = sum(1 for v in scores if v <= lo) / n
    ent = S.entropy_bits(scores)
    declared = (hi - lo + 1) if _is_integral([lo, hi]) else float("nan")

    # --- 1. granularity collapse --------------------------------------------
    sev = SEV_CLEAN
    if eff <= EFFECTIVE_INVALIDATING:
        sev = SEV_INVALIDATING
    elif eff <= EFFECTIVE_SERIOUS:
        sev = SEV_SERIOUS
    elif declared == declared and eff < declared / 2:
        sev = SEV_MINOR

    dist = ", ".join(f"{v:g}×{sum(1 for s in scores if s == v)}" for v in used[:12])
    detail = (f"The judge used {len(used)} distinct value(s): {dist}. Entropy "
              f"{ent:.2f} bits → effective resolution {eff:.1f} level(s).")
    if declared == declared:
        detail += (f"\n   The rubric declares {declared:.0f} levels ({lo:g}-{hi:g}). "
                   f"The judge behaves as though it has {eff:.1f}.")
    if sev in (SEV_SERIOUS, SEV_INVALIDATING):
        detail += ("\n   Everyone reading a score 'out of 10' believes they are looking "
                   "at ten rungs of measurement. They are looking at "
                   f"{eff:.1f}. Differences quoted to two decimals on this scale are "
                   "not measurements, they are rounding.")

    res.findings.append(Finding(
        id="bias.dist.granularity",
        title=f"Judge's effective resolution is {eff:.1f} level(s), not {declared:.0f}"
              if declared == declared else f"Judge's effective resolution is {eff:.1f} level(s)",
        severity=sev,
        confidence=CONF_CERTAIN,   # A histogram. Not an inference.
        detail=detail,
        effect=eff,
        effect_label="effective levels",
        n=n,
        evidence=[f"distinct values used: {len(used)}", f"entropy {ent:.3f} bits",
                  f"distribution: {dist}"],
        fix=("Either use a scale the judge actually spreads across (often a 3-point "
             "rubric with sharp criteria beats an unused 10-point one), or report "
             "the score histogram alongside the mean so nobody over-reads it."),
        why="Perplexity of the score distribution is how many levels the judge behaves "
            "as if it has, regardless of how many the rubric names.",
        tags=["distribution", "granularity"],
    ))

    # --- 2. ceiling ----------------------------------------------------------
    if ceil >= 0.25:
        csev = (SEV_INVALIDATING if ceil >= CEILING_INVALIDATING
                else SEV_SERIOUS if ceil >= CEILING_SERIOUS else SEV_MINOR)
        res.findings.append(Finding(
            id="bias.dist.ceiling",
            title=f"{ceil:.0%} of items already score maximum ({hi:g})",
            severity=csev,
            confidence=CONF_CERTAIN,
            detail=(f"{ceil:.0%} of items sit at the top of the scale. Improvements to "
                    f"those items are unmeasurable — there is nowhere up to go — so "
                    f"this eval will keep reporting 'no significant change' while the "
                    f"system genuinely gets better on {ceil:.0%} of its inputs.\n"
                    f"   Only the remaining {1 - ceil:.0%} can move at all. Any effect "
                    f"you measure is diluted by roughly that factor."),
            effect=ceil,
            effect_label="ceiling share",
            n=n,
            fix="Harden the eval set until the score distribution has room above it.",
            tags=["distribution", "ceiling"],
        ))
    if floor_share >= 0.40:
        res.findings.append(Finding(
            id="bias.dist.floor",
            title=f"{floor_share:.0%} of items score minimum ({lo:g})",
            severity=SEV_SERIOUS if floor_share >= CEILING_INVALIDATING else SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=("A floor effect is the mirror image of a ceiling: regressions on "
                    "these items are invisible, and the eval understates harm."),
            effect=floor_share,
            effect_label="floor share",
            n=n,
            fix="Ease the eval set, or accept that this segment is unmeasured.",
            tags=["distribution", "floor"],
        ))

    # --- 3. dead levels ------------------------------------------------------
    if declared == declared and _is_integral(scores):
        dead = S.unused_levels(scores, lo, hi, 1.0)
        if dead:
            res.findings.append(Finding(
                id="bias.dist.dead-levels",
                title=f"{len(dead)} rubric level(s) never used: {', '.join(f'{d:g}' for d in dead)}",
                severity=SEV_MINOR if len(dead) < declared / 2 else SEV_SERIOUS,
                confidence=CONF_CERTAIN,
                detail=(f"Across {n} judgements the judge never once emitted "
                        f"{', '.join(f'{d:g}' for d in dead)}. Whatever your rubric says "
                        f"those levels mean has never influenced a single score. They "
                        f"are documentation, not measurement."),
                effect=float(len(dead)),
                effect_label="unused levels",
                n=n,
                fix="Cut the scale down to the levels the judge actually uses, and "
                    "write sharper criteria for those.",
                tags=["distribution", "dead-levels"],
            ))

    # --- 4. quantisation: what your delta actually is ------------------------
    if _is_integral(scores) and claimed_delta:
        step = 1.0
        items_hopping = abs(claimed_delta) * n / step
        min_move = step / n
        res.findings.append(Finding(
            id="bias.dist.quantisation",
            title=(f"Your {claimed_delta:+.3f} delta is {items_hopping:.0f} item(s) "
                   f"hopping one notch"),
            severity=SEV_SERIOUS if items_hopping < n * 0.1 else SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"This judge emits whole numbers only. Over {n} items the mean can "
                    f"only move in steps of {min_move:.4f}, so a {claimed_delta:+.3f} "
                    f"change is not {n} items each improving slightly — it is about "
                    f"{items_hopping:.0f} item(s) crossing a single threshold while "
                    f"the other {n - items_hopping:.0f} did not move at all.\n"
                    f"   Whether that is a result depends entirely on whether those "
                    f"{items_hopping:.0f} items were sitting on a boundary the judge "
                    f"is near-coin-flipping on. Quote the mean to two decimals and "
                    f"that question never gets asked."),
            effect=items_hopping,
            effect_label="items moved",
            n=n,
            evidence=[f"minimum resolvable mean change: {min_move:.4f}",
                      f"score granularity: integer"],
            fix=("Report how many items changed level, not just the mean. Better: "
                 "report the full histogram of per-item changes."),
            tags=["distribution", "quantisation"],
        ))
    elif _is_integral(scores):
        res.notes.append(
            f"Judge emits integers only. Over {n} items the mean cannot move by less "
            f"than {1.0 / n:.4f} — pass claimed_delta to see what a given headline "
            f"figure means in items-that-actually-moved.")

    # --- 5. constant judge ---------------------------------------------------
    if len(used) == 1:
        res.findings.append(Finding(
            id="bias.dist.constant",
            title=f"The judge emitted {used[0]:g} for every single item",
            severity=SEV_INVALIDATING,
            confidence=CONF_CERTAIN,
            detail=("Zero variance. This judge has not discriminated between any two "
                    "outputs in the entire run and carries no information whatsoever. "
                    "Note that κ against a constant rater is degenerate too, so a "
                    "calibration run on this judge may report a flattering number — "
                    "this finding is the reason to distrust it."),
            n=n,
            fix="Check the judge is actually being called and its output parsed. A "
                "constant score is usually a parsing bug, not a judgement.",
            tags=["distribution", "constant"],
        ))

    return res
