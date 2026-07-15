"""calibrate_judge — does the judge agree with humans, beyond chance?

The question this answers is not "is the judge accurate". Accuracy is the trap.
On a benchmark where 78% of items pass, a judge that has learned nothing except
the phrase "pass" scores 78%, and 78% reads as a good number to everyone who
sees it in a slide. Cohen's κ subtracts the luck out, and the gap between those
two numbers is the finding.

Three things this does that a bare κ does not:

1. Reports the majority-class baseline next to accuracy, always. The baseline is
   what makes 81% legible as "3 points better than a constant".

2. Checks the human ceiling. If your annotators only agree with each other at
   κ=0.55, a judge at κ=0.50 is at the noise floor of the task and "improve the
   judge" is the wrong project — the rubric is underdefined and no judge, human
   or otherwise, can do better. Tools that skip this send teams on month-long
   goose chases to fix a judge that is already as good as a person.

3. Refuses to over-claim from small n. κ from 14 items is not a measurement.

The failure this is aimed at: nobody has *ever* checked. Judges ship uncalibrated
because calibration requires human labels, human labels cost money, and the judge
produces plausible-looking numbers without them.
"""
from __future__ import annotations

from typing import Optional

from . import stats as S
from .io import consensus_fingerprint, load_gold_labels, load_judge_records, pair_by_item
from .models import (
    CONF_CERTAIN,
    CONF_LIKELY,
    POWER_FLOOR_KAPPA,
    AuditResult,
    Finding,
    GoldLabel,
    JudgeRecord,
    SEV_CLEAN,
    SEV_INVALIDATING,
    SEV_MINOR,
    SEV_SERIOUS,
    kappa_band,
    temper_confidence,
)

# κ thresholds for severity. Landis & Koch bands are a convention, not a law, so
# these are stated as a judgement call rather than smuggled in as arithmetic:
# below 0.4 a judge disagrees with humans often enough that per-item verdicts are
# not trustworthy; below 0.2 it is barely distinguishable from a base-rate guess
# and any conclusion drawn from it is unsupported.
KAPPA_SERIOUS = 0.40
KAPPA_INVALIDATING = 0.20

# How close to the human ceiling counts as "at the ceiling". Within 0.1 of
# inter-human κ means the judge is inside the noise band of the task itself.
CEILING_MARGIN = 0.10


def _discretise(pairs: list[tuple[JudgeRecord, GoldLabel]]) -> tuple[list, list, str]:
    """Extract comparable rater vectors and say which axis we used.

    Priority: explicit labels > pairwise choices > numeric scores. Labels first
    because if a human wrote "fail" and the judge wrote "fail", that's the
    comparison the eval actually cares about; the numeric score is a proxy for it.
    """
    if all(j.label is not None and g.label is not None for j, g in pairs):
        return ([j.label for j, _ in pairs], [g.label for _, g in pairs], "label")
    if all(j.choice is not None and g.choice is not None for j, g in pairs):
        return ([j.choice for j, _ in pairs], [g.choice for _, g in pairs], "choice")
    js = [j.score for j, _ in pairs]
    gs = [g.score for _, g in pairs]
    if all(v is not None for v in js) and all(v is not None for v in gs):
        return (js, gs, "score")
    return ([], [], "none")


def _ordinal(vals: list) -> bool:
    return all(isinstance(v, (int, float)) for v in vals)


def calibrate(
    judge_path: str,
    gold_path: str,
    weights: str = "auto",
    human_ceiling: Optional[float] = None,
) -> AuditResult:
    """Compute agreement between a judge run and a human-labelled set."""
    judge, jrep = load_judge_records(judge_path)
    gold, grep = load_gold_labels(gold_path)
    pairs, warns = pair_by_item(judge, gold)

    res = AuditResult(kind="calibration")
    res.notes.extend(jrep.warnings + grep.warnings + warns)

    if not pairs:
        res.findings.append(Finding(
            id="cal.no-overlap",
            title="No item_id appears in both the judge run and the human labels",
            severity=SEV_INVALIDATING,
            confidence=CONF_CERTAIN,
            detail="These two files describe different populations, so no agreement "
                   "statistic is defined. Check that item_id means the same thing "
                   "in both files.",
            fix="Align item_id across the judge log and the label file.",
        ))
        return res

    a, b, axis = _discretise(pairs)
    res.n_items = len(pairs)
    res.fingerprint, _ = consensus_fingerprint([j for j, _ in pairs])

    if axis == "none":
        res.findings.append(Finding(
            id="cal.no-axis",
            title="Judge and humans have no comparable verdict field",
            severity=SEV_INVALIDATING,
            confidence=CONF_CERTAIN,
            detail="Neither labels, pairwise choices, nor numeric scores are present "
                   "on both sides for every paired item.",
            fix="Ensure both files carry the same kind of verdict.",
        ))
        return res

    n = len(a)

    # --- the headline: κ, with the baseline it must be read against ----------
    ordinal = axis == "score" and _ordinal(a) and _ordinal(b)
    use_weighted = ordinal and (weights in ("linear", "quadratic")
                                or (weights == "auto" and len(set(a) | set(b)) > 2))
    wt = weights if weights in ("linear", "quadratic") else "linear"

    if use_weighted:
        k = S.weighted_kappa(a, b, wt)
        kname = f"weighted κ ({wt})"
        ci = S.bootstrap_ci(lambda idx: S.weighted_kappa([a[i] for i in idx],
                                                         [b[i] for i in idx], wt), n)
    else:
        k = S.cohens_kappa(a, b)
        kname = "Cohen's κ"
        ci = S.bootstrap_ci(lambda idx: S.cohens_kappa([a[i] for i in idx],
                                                       [b[i] for i in idx]), n)

    acc = S.observed_agreement(a, b)
    base, top = S.majority_baseline(b)
    pe = S.chance_agreement(a, b)

    sev = SEV_CLEAN
    if k < KAPPA_INVALIDATING:
        sev = SEV_INVALIDATING
    elif k < KAPPA_SERIOUS:
        sev = SEV_SERIOUS
    elif k < 0.61:
        sev = SEV_MINOR

    conf = temper_confidence(CONF_LIKELY, n, POWER_FLOOR_KAPPA)

    detail = (f"Raw agreement {acc:.1%}, but {pe:.1%} of that is expected by chance "
              f"given the base rates. {kname} = {k:.3f} ({kappa_band(k)}).")
    if base == base:
        detail += (f" A judge that always said {top!r} would score {base:.1%} "
                   f"accuracy on this set.")

    res.findings.append(Finding(
        id="cal.kappa",
        title=f"Judge–human agreement: {kname} = {k:.3f} ({kappa_band(k)})",
        severity=sev,
        confidence=conf,
        detail=detail,
        effect=k,
        effect_label=kname,
        ci_low=ci[0],
        ci_high=ci[1],
        n=n,
        evidence=[f"axis: {axis}",
                  f"observed agreement {acc:.3f}",
                  f"chance agreement {pe:.3f}",
                  f"majority baseline {base:.3f} (always {top!r})"],
        fix=("Below κ≈0.4 the judge's per-item verdicts disagree with humans too "
             "often to support per-item claims. Either tighten the rubric until "
             "humans and judge converge, or stop reporting this judge's scores "
             "as if they measured quality."),
        why=("κ is agreement corrected for chance. Accuracy is not, which is why "
             "an unconditional 'pass' scores well on any realistic eval set."),
        citation="Cohen (1960); bands per Landis & Koch (1977) — a convention, not a law.",
        tags=["calibration", "kappa"],
    ))

    # --- the baseline finding, when accuracy is doing the flattering ---------
    if base == base and acc - base < 0.10:
        res.findings.append(Finding(
            id="cal.baseline",
            title=f"Judge beats a constant-answer baseline by only {acc - base:+.1%}",
            severity=SEV_SERIOUS if acc - base < 0.05 else SEV_MINOR,
            confidence=temper_confidence(CONF_LIKELY, n, POWER_FLOOR_KAPPA),
            detail=(f"{acc:.1%} accuracy sounds respectable until you notice that "
                    f"always answering {top!r} scores {base:.1%} on this set. The "
                    f"judge is contributing {acc - base:+.1%} over a rule that reads "
                    f"none of the input."),
            effect=acc - base,
            effect_label="delta vs majority baseline",
            n=n,
            evidence=["prevalence: " + ", ".join(f"{k2!r}={v:.1%}"
                                                 for k2, v in S.prevalence(b).items())],
            fix="Rebalance the eval set, or report κ instead of accuracy.",
            tags=["calibration", "baseline"],
        ))

    # --- the human ceiling: is the judge bad, or is the task underdefined? ---
    ceil = human_ceiling
    if ceil is None:
        agrees = [g.rater_agreement for _, g in pairs if g.rater_agreement is not None]
        if agrees:
            ceil = S.mean(agrees)

    if ceil is not None:
        if k >= ceil - CEILING_MARGIN:
            res.findings.append(Finding(
                id="cal.ceiling",
                title=f"Judge is at the human ceiling (κ={k:.3f} vs inter-human κ={ceil:.3f})",
                severity=SEV_CLEAN,
                confidence=temper_confidence(CONF_LIKELY, n, POWER_FLOOR_KAPPA),
                detail=("Your annotators agree with each other about as well as the "
                        "judge agrees with them. The judge is not the bottleneck — "
                        "the task is. No judge can exceed the reliability of the "
                        "labels it is scored against."),
                effect=k - ceil,
                effect_label="delta vs human ceiling",
                n=n,
                fix=("If you need higher agreement, sharpen the rubric until humans "
                     "converge. Replacing the judge will not move this number."),
                why="κ against noisy labels is bounded above by the noise in those labels.",
                tags=["calibration", "ceiling"],
            ))
        else:
            res.findings.append(Finding(
                id="cal.below-ceiling",
                title=f"Judge is {ceil - k:.3f} below the human ceiling",
                severity=SEV_MINOR,
                confidence=temper_confidence(CONF_LIKELY, n, POWER_FLOOR_KAPPA),
                detail=(f"Humans agree with each other at κ={ceil:.3f}; the judge "
                        f"reaches κ={k:.3f}. There is real headroom here — this gap "
                        f"belongs to the judge, not to the task."),
                effect=k - ceil,
                effect_label="delta vs human ceiling",
                n=n,
                fix="Prompt/rubric work on the judge should close this.",
                tags=["calibration", "ceiling"],
            ))
    else:
        res.notes.append(
            "No inter-human agreement recorded, so the ceiling is unknown — we can't "
            "tell you whether a mediocre κ means a bad judge or an underdefined task. "
            "Record `rater_agreement` in the label file, or pass human_ceiling.")

    # --- power ---------------------------------------------------------------
    if n < POWER_FLOOR_KAPPA:
        res.findings.append(Finding(
            id="cal.underpowered",
            title=f"Only {n} labelled item(s) — κ here is an anecdote",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"κ's sampling variance is large at this n, and larger still when "
                    f"one class dominates. The interval [{ci[0]:.2f}, {ci[1]:.2f}] is "
                    f"the honest summary; the point estimate is not. Statistical "
                    f"findings in this report are marked heuristic and will not gate CI."),
            n=n,
            fix=f"Label at least {POWER_FLOOR_KAPPA} items, ideally {POWER_FLOOR_KAPPA * 2}.",
            tags=["power"],
        ))

    # --- correlation, for numeric axes --------------------------------------
    if ordinal:
        r = S.pearson_r(a, b)
        if r == r:
            res.findings.append(Finding(
                id="cal.correlation",
                title=f"Judge–human score correlation r = {r:.3f}",
                severity=SEV_CLEAN if r >= 0.7 else SEV_MINOR,
                confidence=temper_confidence(CONF_LIKELY, n, POWER_FLOOR_KAPPA),
                detail=("Reported for context only. It is a weaker claim than κ: a "
                        "judge that scores every item exactly 3 points too high "
                        "correlates perfectly and agrees with nobody."),
                effect=r,
                effect_label="Pearson r",
                n=n,
                tags=["calibration", "correlation"],
            ))

    return res
