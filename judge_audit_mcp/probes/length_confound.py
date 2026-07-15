"""Length vs quality — is the judge rewarding length, or is length just correlated
with quality?

This probe exists to survive the obvious objection to `verbosity`.

Someone will say: "Of course the judge scores long answers higher. Long answers
*are* better — they're more thorough. Your 'bias' is the judge being right." And
on observational data, that person cannot be refuted by a correlation. Length and
quality really are correlated in real outputs. A judge that scored them equally
would be the broken one.

The fix is to control for quality using the human score, and ask the only
question that separates the two stories:

    judge_score ~ b0 + b1·human_score + b2·length

`b2` is the score the judge pays for length **among answers humans rated
equally**. Under "length is a proxy for quality", b2 ≈ 0 — once you know how good
an answer is, its length tells the judge nothing more, and there's nothing left
for it to pay for. Under "the judge likes long answers", b2 > 0.

That's the entire argument, and it's why this probe is worth more than the naive
correlation it replaces. The naive `r(length, judge_score)` is consistent with
both stories and therefore evidence for neither.

Bivariate `r(length, judge_score)` is still reported — as the number people
usually quote, next to the number that actually answers the question, so the gap
between them is visible.

Honest limits, stated because the whole probe is an argument about confounding
and it would be graceless to hide its own:

  - `human_score` is assumed to capture quality. If your annotators are themselves
    length-biased, b2 under-reports. This measures the judge's excess bias *over
    the humans'*, not bias in some absolute sense.
  - Linearity in length is an approximation. Real length effects plausibly
    saturate. b2 is an average slope over the observed range, not a law.
  - Observational. Nothing was randomised, so unmeasured things correlated with
    both length and judge score could carry some of b2. `verbosity` is the
    interventional complement, and the two together are much stronger than either
    alone: one has clean identification on constructed data, the other has real
    data with a confound controlled.
"""
from __future__ import annotations

from .. import stats as S
from ..io import load_gold_labels, load_judge_records, pair_by_item
from ..models import (
    CONF_CERTAIN,
    CONF_LIKELY,
    POWER_FLOOR_PAIRED,
    AuditResult,
    Finding,
    SEV_CLEAN,
    SEV_MINOR,
    SEV_SERIOUS,
    temper_confidence,
)

# Points bought per 1000 characters, as a fraction of the judge's used range.
SLOPE_SERIOUS = 0.05
SLOPE_INVALIDATING = 0.15


def probe_length_confound(path: str, gold_path: str = "", **kw) -> AuditResult:
    recs, rep = load_judge_records(path)
    res = AuditResult(kind="bias")
    res.notes.extend(rep.warnings)
    if recs:
        res.fingerprint = recs[0].fingerprint

    if not gold_path:
        xs = [float(r.length()) for r in recs if r.score is not None and r.length() > 0]
        ys = [r.score for r in recs if r.score is not None and r.length() > 0]
        r = S.pearson_r(xs, ys) if len(xs) > 2 else float("nan")
        res.findings.append(Finding(
            id="bias.length.no-gold",
            title="No human scores — length effect cannot be separated from quality",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"Bivariate correlation between length and judge score: r={r:.3f}. "
                    f"This number is famous and worthless. It is exactly as consistent "
                    f"with 'the judge is length-biased' as with 'longer answers are "
                    f"better and the judge noticed'. Separating those needs a quality "
                    f"control, and the human score is it."),
            effect=r,
            effect_label="Pearson r (confounded)",
            n=len(xs),
            fix="Supply gold_path with human scores to fit the controlled model.",
            tags=["length_confound", "confounded"],
        ))
        return res

    gold, grep = load_gold_labels(gold_path)
    pairs, warns = pair_by_item(recs, gold)
    res.notes.extend(grep.warnings + warns)

    X, y, lens = [], [], []
    for rec, g in pairs:
        if rec.score is None or g.score is None:
            continue
        L = rec.length()
        if L <= 0:
            continue
        X.append([float(g.score), L / 1000.0])   # length in kilochars: b2 is per-1k
        y.append(float(rec.score))
        lens.append(float(L))

    n = len(y)
    res.n_items = n
    if n < 10:
        res.findings.append(Finding(
            id="bias.length.thin",
            title=f"Only {n} item(s) with score, human score and length",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail="Not enough to fit a two-predictor model. Ensure records carry "
                   "output_text or output_len.",
            n=n,
            tags=["length_confound", "power"],
        ))
        return res

    beta = S.ols(X, y)
    if beta is None:
        res.findings.append(Finding(
            id="bias.length.singular",
            title="Regression is singular — length and human score are collinear here",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=("Length and human score carry the same information in this sample, "
                    "so their effects cannot be separated. Usually means the eval set "
                    "has no long-but-bad or short-but-good answers — which is itself "
                    "worth knowing, since it's the design a length-biased judge would "
                    "pass."),
            n=n,
            fix="Include items that break the length/quality correlation.",
            tags=["length_confound"],
        ))
        return res

    b_human, b_len = beta[1], beta[2]
    ci = S.ols_slope_ci(X, y, coef=2)
    r2 = S.r_squared(X, y, beta)
    r_naive = S.pearson_r(lens, y)
    r_human_len = S.pearson_r([row[0] for row in X], lens)

    rng = (max(y) - min(y)) if y else 0.0
    spread_k = (max(lens) - min(lens)) / 1000.0
    across = b_len * spread_k              # points bought across the observed length range
    rel = abs(across) / rng if rng else float("nan")

    sig = ci[0] == ci[0] and (ci[0] > 0 or ci[1] < 0)
    sev = SEV_CLEAN
    if sig and rel == rel:
        if rel >= SLOPE_INVALIDATING:
            sev = SEV_SERIOUS
        elif rel >= SLOPE_SERIOUS:
            sev = SEV_MINOR
    conf = temper_confidence(CONF_LIKELY if sig else "heuristic", n, POWER_FLOOR_PAIRED)

    detail = (
        f"Naive correlation r(length, judge score) = {r_naive:.3f} — the number "
        f"usually quoted, and it settles nothing.\n"
        f"   Controlling for human score:\n"
        f"     judge ≈ {beta[0]:.2f} + {b_human:.3f}·human + {b_len:.3f}·(length/1k)\n"
        f"   Among answers humans rated *equally*, every extra 1000 characters buys "
        f"{b_len:+.3f} judge points (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], n={n}, "
        f"R²={r2:.2f}).\n"
        f"   Across the length range actually present here ({spread_k:.1f}k chars), "
        f"that is {across:+.2f} points of pure length premium."
    )
    if sig and b_len > 0:
        detail += ("\n   The 'longer answers are simply better' defence does not hold: "
                   "quality is already in the model, and length is still paid for on "
                   "top of it.")
    elif not sig:
        detail += ("\n   Once quality is controlled, length buys nothing detectable. "
                   f"The naive r={r_naive:.2f} is length tracking quality, not the "
                   f"judge chasing length — this judge is exonerated on this axis.")

    res.findings.append(Finding(
        id="bias.length.controlled",
        title=(f"Length buys {b_len:+.3f} points per 1k chars at equal human quality"
               if sig else "No length premium once quality is controlled"),
        severity=sev,
        confidence=conf,
        detail=detail,
        effect=b_len,
        effect_label="delta (judge points per 1k chars, human-controlled)",
        ci_low=ci[0],
        ci_high=ci[1],
        n=n,
        evidence=[
            f"naive r(length, judge)        {r_naive:+.3f}",
            f"r(human score, length)        {r_human_len:+.3f}  ← the confound, sized",
            f"b_human                       {b_human:+.3f}",
            f"b_length (per 1k chars)       {b_len:+.3f}  CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
            f"model R²                      {r2:.3f}",
        ],
        fix=("If the premium is real, length-normalise before judging, or say "
             "explicitly in the rubric that length is not a criterion and re-probe."),
        why=("Controlling for human score is what turns a correlation into a claim "
             "about the judge. Without it, the two stories are observationally "
             "identical."),
        citation="Dubois et al. (2024), 'Length-Controlled AlpacaEval'",
        tags=["length_confound"],
    ))
    return res
