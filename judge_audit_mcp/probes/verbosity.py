"""Verbosity bias — does padding an answer make the judge like it more?

The manipulation: take an output, add filler that carries no information ("It's
worth noting that...", a restated conclusion, a bulleted recap of what was just
said), and re-judge. The answer is not better. It is only longer.

Any score lift is bias, and it is expensive bias, because it is the one your
system will *learn*. A judge that pays for length is a judge your prompt-tuning
loop will discover and exploit within a week — you'll ship a model that pads,
your eval score will rise, and your users will get more words and no more value.
Optimising against a biased judge doesn't just mismeasure the system, it deforms it.

The honest caveat, and the reason `length_confound` exists as a separate probe:
this probe only licenses "the judge rewards padding *on these items*". Someone
will object that longer answers really are better in general, so a judge that
scores them higher is right, not biased. That objection is wrong *here* — padding
is quality-preserving by construction — but it's a fair objection to the naive
observational version of this question, and `length_confound` is the probe that
answers it on real data where nothing was constructed.
"""
from __future__ import annotations

from .. import stats as S
from ..io import load_judge_records
from ..models import (
    CONF_CERTAIN,
    CONF_LIKELY,
    POWER_FLOOR_PAIRED,
    AuditResult,
    Finding,
    SEV_CLEAN,
    SEV_INVALIDATING,
    SEV_MINOR,
    SEV_SERIOUS,
    temper_confidence,
)

# Score lift, as a fraction of the observed score range, above which padding is
# buying enough to change decisions.
LIFT_SERIOUS = 0.05      # 5% of range
LIFT_INVALIDATING = 0.15


def probe_verbosity(path: str, **kw) -> AuditResult:
    """Expects two records per item_id: variant "" and variant "padded"."""
    recs, rep = load_judge_records(path)
    res = AuditResult(kind="bias")
    res.notes.extend(rep.warnings)

    orig = {r.item_id: r for r in recs if r.variant in ("", "original")}
    pad = {r.item_id: r for r in recs if r.variant in ("padded", "pad", "verbose")}
    ids = sorted(set(orig) & set(pad))

    if not ids:
        res.findings.append(Finding(
            id="bias.verbosity.no-variants",
            title="No padded variants found — verbosity bias cannot be probed",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=("This probe needs each output judged twice: as written, and padded "
                    "with filler that adds no information, tagged variant='padded' "
                    "under the same item_id."),
            fix="Use `emit_probe_set(probe='verbosity')` to generate the padded file.",
            tags=["verbosity"],
        ))
        return res

    res.fingerprint = recs[0].fingerprint
    res.n_items = len(ids)

    deltas, growth = [], []
    for i in ids:
        a, b = orig[i], pad[i]
        if a.score is None or b.score is None:
            continue
        deltas.append(b.score - a.score)
        if a.length() > 0:
            growth.append(b.length() / a.length())

    n = len(deltas)
    if n == 0:
        res.notes.append("No scored pairs.")
        return res

    lift = S.mean(deltas)
    ci = S.mean_delta_ci(deltas)
    p, n_pos, n_nz = S.sign_test(deltas)
    d = S.cohens_d_paired(deltas)

    all_scores = [r.score for r in recs if r.score is not None]
    rng = (max(all_scores) - min(all_scores)) if all_scores else 0.0
    rel = abs(lift) / rng if rng else float("nan")

    sig = ci[0] == ci[0] and (ci[0] > 0 or ci[1] < 0)
    sev = SEV_CLEAN
    if sig and rel == rel:
        if rel >= LIFT_INVALIDATING:
            sev = SEV_INVALIDATING
        elif rel >= LIFT_SERIOUS:
            sev = SEV_SERIOUS
        else:
            sev = SEV_MINOR
    conf = temper_confidence(CONF_LIKELY if sig else "heuristic", n, POWER_FLOOR_PAIRED)

    mean_growth = S.mean(growth) if growth else float("nan")
    detail = (f"Padding the same answers with filler moved the judge {lift:+.3f} "
              f"(95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], n={n}). "
              f"{n_pos}/{n_nz} padded answers scored higher, p={p:.3g}.")
    if mean_growth == mean_growth:
        detail += f" Padding made outputs {mean_growth:.1f}× longer on average."
    if rng:
        detail += (f"\n   That lift is {rel:.1%} of the {rng:.0f}-point range the judge "
                   f"actually uses.")
    if sig and lift > 0:
        detail += ("\n   Nothing about these answers improved. The judge is paying for "
                   "length, and any optimiser pointed at this judge will find that out "
                   "faster than you will.")

    res.findings.append(Finding(
        id="bias.verbosity.lift",
        title=(f"Padding with filler buys {lift:+.3f} points"
               if sig else f"No detectable verbosity lift ({lift:+.3f})"),
        severity=sev,
        confidence=conf,
        detail=detail,
        effect=lift,
        effect_label="delta (padded − original)",
        ci_low=ci[0],
        ci_high=ci[1],
        n=n,
        p_value=p,
        evidence=[f"paired Cohen's d {d:.2f}" if d == d else "paired Cohen's d n/a",
                  f"mean length growth {mean_growth:.2f}×" if mean_growth == mean_growth else "",
                  f"score range used by judge: {rng:.0f}"],
        fix=("Add an explicit length-neutrality instruction to the judge prompt and "
             "re-probe — this is one of the few biases that often responds to prompt "
             "work. If it persists, normalise for length or truncate before judging."),
        why=("Filler is quality-preserving by construction, so any score response to "
             "it is bias with no defence available."),
        citation="Zheng et al. (2023); Dubois et al. (2024), 'Length-Controlled AlpacaEval'",
        tags=["verbosity"],
    ))
    return res
