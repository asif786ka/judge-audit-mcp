"""Self-preference — does the judge mark its own family's homework generously?

Why this one needs a human panel, and why that's the whole design.

The naive version of this probe compares the judge's mean score for Claude
outputs against its mean for GPT outputs, finds a gap, and calls it bias. That
inference is invalid, and confidently so: maybe one model just wrote better
answers. You cannot distinguish "the judge favours its own" from "its own was
actually better" by looking at judge scores alone. No amount of data fixes this —
it's a confound, not a sample size problem.

So this probe measures the **residual against a human panel**:

    residual(family) = mean(judge_score − human_score)  over that family's outputs

The human score absorbs real quality differences. What's left is the judge's
disagreement with humans, per family. If the judge over-scores its own family by
+0.81 relative to humans, and under-scores others by −0.12, the 0.93 gap is not
explainable by quality — humans already priced quality in.

Where this bites hardest: model bakeoffs. The judge is usually whatever's best
and cheapest, which is usually from the same family as one of the candidates.
That's a rigged comparison, run in good faith, published as a benchmark.

Panickssery et al. (2024) find LLMs recognise and favour their own generations.
The effect is modest — well under a point on a 10-point scale — which is exactly
why this probe reports intervals rather than a verdict: an effect that size is
trivially manufactured by n=20 and a hopeful analyst.
"""
from __future__ import annotations

from typing import Optional

from .. import stats as S
from ..io import load_gold_labels, load_judge_records, pair_by_item
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

# Self-preference delta as a fraction of the judge's used score range.
DELTA_SERIOUS = 0.05
DELTA_INVALIDATING = 0.12


def _family_of(model: str) -> str:
    """Coarse family from a model string. Deliberately crude — the alternative is
    a vendor table that rots, and the caller can always set output_family."""
    m = (model or "").lower()
    for fam, keys in (
        ("claude", ("claude", "anthropic", "sonnet", "opus", "haiku")),
        ("gpt", ("gpt", "openai", "o1", "o3", "davinci", "chatgpt")),
        ("gemini", ("gemini", "google", "bard", "palm")),
        ("llama", ("llama", "meta")),
        ("mistral", ("mistral", "mixtral")),
        ("qwen", ("qwen", "alibaba")),
        ("deepseek", ("deepseek",)),
        ("grok", ("grok", "xai")),
    ):
        if any(k in m for k in keys):
            return fam
    return ""


def probe_self_preference(path: str, gold_path: str = "",
                          judge_family: str = "", **kw) -> AuditResult:
    """Needs `output_family` on records and a human-labelled gold file."""
    recs, rep = load_judge_records(path)
    res = AuditResult(kind="bias")
    res.notes.extend(rep.warnings)

    if not recs:
        return res
    res.fingerprint = recs[0].fingerprint

    jf = (judge_family or _family_of(res.fingerprint.model)).lower()
    if not jf:
        res.findings.append(Finding(
            id="bias.self.no-judge-family",
            title="Can't tell which family the judge belongs to",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=("Self-preference is a claim about the judge scoring *its own* "
                    "family generously, so we need to know the judge's family. No "
                    "judge model string was recorded and none was passed."),
            fix="Pass judge_family='claude' (or log judge_model in the eval).",
            tags=["self_preference"],
        ))
        return res

    fams = {(r.output_family or "").lower() for r in recs if r.output_family}
    if len(fams) < 2:
        res.findings.append(Finding(
            id="bias.self.one-family",
            title="Only one output family present — nothing to compare",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"Self-preference is comparative. Found: {', '.join(sorted(fams)) or 'none'}. "
                    f"Needs outputs from at least two families, one of them {jf!r}."),
            fix="Tag each record with output_family, including non-judge-family outputs.",
            tags=["self_preference"],
        ))
        return res

    if not gold_path:
        judge_only = {f: S.mean([r.score for r in recs
                                 if (r.output_family or "").lower() == f and r.score is not None])
                      for f in sorted(fams)}
        gap = judge_only.get(jf, float("nan")) - S.mean(
            [v for f, v in judge_only.items() if f != jf] or [float("nan")])
        res.findings.append(Finding(
            id="bias.self.no-gold",
            title="No human panel — a raw score gap cannot establish self-preference",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"The judge scores {jf!r} outputs {gap:+.3f} above other families. "
                    f"That number is *not* evidence of bias and must not be quoted as "
                    f"such: the {jf!r} outputs may simply be better. Judge scores alone "
                    f"cannot separate 'favours its own' from 'its own was better' — "
                    f"the confound is structural, and more data will not resolve it.\n"
                    f"   Per family: " + ", ".join(f"{f}={v:.2f}" for f, v in judge_only.items())),
            effect=gap,
            effect_label="delta (raw, confounded)",
            n=len(recs),
            fix=("Supply a human-labelled gold file. The residual against humans is "
                 "the only version of this measurement that means anything."),
            tags=["self_preference", "confounded"],
        ))
        return res

    gold, grep = load_gold_labels(gold_path)
    pairs, warns = pair_by_item(recs, gold)
    res.notes.extend(grep.warnings + warns)
    res.n_items = len(pairs)

    by_fam: dict[str, list[float]] = {}
    for r, g in pairs:
        if r.score is None or g.score is None or not r.output_family:
            continue
        by_fam.setdefault(r.output_family.lower(), []).append(r.score - g.score)

    if jf not in by_fam:
        # Not a limitation — the recommended configuration. A judge with no
        # candidate of its own in the race cannot self-prefer; there is nothing
        # for the bias to attach to. Reporting this as "couldn't measure" would
        # be exactly backwards, and would nudge people away from the one setup
        # that makes the whole problem go away.
        spread = {f: S.mean(ds) for f, ds in sorted(by_fam.items(),
                                                    key=lambda kv: -S.mean(kv[1]))}
        gap = max(spread.values()) - min(spread.values()) if len(spread) > 1 else 0.0
        res.findings.append(Finding(
            id="bias.self.uninvolved-judge",
            title=f"Judge ({jf}) has no output of its own family in this comparison",
            severity=SEV_CLEAN,
            confidence=CONF_CERTAIN,
            detail=(f"Self-preference is structurally impossible here: the judge is "
                    f"{jf!r} and the candidates are {', '.join(sorted(by_fam))}. There "
                    f"is no own-family output for the bias to attach to.\n"
                    f"   This is the recommended setup, not a gap in the data. A judge "
                    f"with no dog in the fight cannot mark its own homework."),
            n=len(pairs),
            evidence=[f"{f}: {v:+.3f} vs humans" for f, v in spread.items()],
            why=("Note this does not rule out the judge favouring some *other* family "
                 f"— the residual spread across families here is {gap:.3f}. That would "
                 "be a different bias with a different cause, and this probe does not "
                 "test for it."),
            tags=["self_preference", "uninvolved"],
        ))
        return res

    own = by_fam[jf]
    others = [d for f, ds in by_fam.items() if f != jf for d in ds]
    if not others:
        res.notes.append("No non-judge-family outputs after pairing.")
        return res

    own_r, oth_r = S.mean(own), S.mean(others)
    delta = own_r - oth_r

    # Unpaired: bootstrap the two-sample difference of residual means.
    def stat(idx):
        n_own = len(own)
        o = [own[i % n_own] for i in idx[:n_own]]
        t = [others[i % len(others)] for i in idx[n_own:]] or others
        return S.mean(o) - S.mean(t)

    ci = S.bootstrap_ci(stat, len(own) + len(others))
    n = min(len(own), len(others))

    all_scores = [r.score for r, _ in pairs if r.score is not None]
    rng = (max(all_scores) - min(all_scores)) if all_scores else 0.0
    rel = abs(delta) / rng if rng else float("nan")

    sig = ci[0] == ci[0] and (ci[0] > 0 or ci[1] < 0)
    sev = SEV_CLEAN
    if sig and rel == rel:
        if rel >= DELTA_INVALIDATING:
            sev = SEV_INVALIDATING
        elif rel >= DELTA_SERIOUS:
            sev = SEV_SERIOUS
        else:
            sev = SEV_MINOR
    conf = temper_confidence(CONF_LIKELY if sig else "heuristic", n, POWER_FLOOR_PAIRED)

    fam_lines = [f"{f}: {S.mean(ds):+.3f} vs humans (n={len(ds)})"
                 for f, ds in sorted(by_fam.items(), key=lambda kv: -S.mean(kv[1]))]

    detail = (f"Measured as disagreement with the human panel, per family — so real "
              f"quality differences are already priced in.\n"
              f"   The judge ({jf}) scores {jf!r} outputs {own_r:+.3f} relative to "
              f"humans, and other families {oth_r:+.3f}. Self-preference delta "
              f"{delta:+.3f} (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]).")
    if sig and delta > 0:
        detail += (f"\n   Humans do not see this gap; only the judge does. Any bakeoff "
                   f"scored by a {jf} judge hands {jf} a {delta:.2f}-point head start "
                   f"that has nothing to do with the outputs.")

    res.findings.append(Finding(
        id="bias.self.delta",
        title=(f"Judge favours its own family by {delta:+.3f} vs humans"
               if sig else f"No detectable self-preference ({delta:+.3f})"),
        severity=sev,
        confidence=conf,
        detail=detail,
        effect=delta,
        effect_label="delta (own-family residual − other-family residual)",
        ci_low=ci[0],
        ci_high=ci[1],
        n=n,
        evidence=fam_lines,
        fix=("Use a judge from a family with no candidate in the race, or an ensemble "
             "across families, and report the spread between them."),
        why=("Residuals against humans absorb genuine quality differences, which is "
             "what makes this a bias claim rather than a quality observation."),
        citation="Panickssery et al. (2024), 'LLM Evaluators Recognize and Favor Their Own Generations'",
        tags=["self_preference"],
    ))
    return res
