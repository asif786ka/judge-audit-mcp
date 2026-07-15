"""detect_judge_drift — did your system improve, or did the ruler move?

This is the tool the server exists for.

Two eval runs. Scores went from 7.21 to 7.64, a 6% improvement, and it goes in
the launch doc. Nobody records that the judge model was bumped between the runs,
because the judge is infrastructure — it's not the thing under test, so its
version isn't in the changelog of the thing under test. The 6% is a measurement
of the judge upgrade. The system never moved.

This failure is silent (no error, no warning, the number is *plausible*),
universal (everyone bumps judge models; the alternative is pinning a deprecated
model forever), and embarrassing when found (the improvement you shipped, told
your team about, and wrote a doc about was an artefact of your measuring
instrument).

## How the decomposition works, and why it needs anchors

You cannot attribute a score change by comparing two means. `mean(B) - mean(A)`
confounds three things:

    1. the system got better        (what you want to measure)
    2. the eval set changed         (items added/removed between runs)
    3. the judge changed            (the ruler moved)

Confound 2 is handled by joining on item_id — compare only items in both runs.

Confound 3 is the hard one, and it needs an **anchor set**: items whose output is
*byte-identical* across the two runs. For those items the system provably did not
change, so any movement in their scores is the judge, wholly and by construction.
That's not an assumption, it's an identity. Measure the judge's shift on the
anchors, subtract, and what's left is the system.

    judge_shift  = mean(score_B - score_A)  over anchors        [system held fixed]
    total_shift  = mean(score_B) - mean(score_A)  over common items
    system_shift = total_shift - judge_shift

Without anchors, this tool will tell you the fingerprint changed and then refuse
to apportion the delta, because apportioning it would require assuming the very
thing in question. That refusal is the honest answer, and the fix is cheap:

    **Freeze ~30 items and their outputs. Re-judge them every run. That is your
    judge canary, and it costs about a dollar.**

Nobody does this, which is why nobody catches judge drift.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from . import stats as S
from .io import consensus_fingerprint, load_judge_records, pair_runs
from .models import (
    CONF_CERTAIN,
    CONF_LIKELY,
    POWER_FLOOR_PAIRED,
    AuditResult,
    Finding,
    JudgeRecord,
    SEV_CLEAN,
    SEV_INVALIDATING,
    SEV_MINOR,
    SEV_SERIOUS,
    temper_confidence,
)

# Share of the headline delta explained by the judge, above which the headline is
# not a result. 0.5 = the judge accounts for more of your "improvement" than your
# system does.
JUDGE_SHARE_INVALIDATING = 0.50
JUDGE_SHARE_SERIOUS = 0.25


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _find_anchors(
    pairs: list[tuple[JudgeRecord, JudgeRecord]],
    anchor_ids: Optional[set] = None,
) -> tuple[list[tuple[JudgeRecord, JudgeRecord]], str]:
    """Items where the system provably did not change between runs.

    Two ways to establish that, in order of strength:

      "declared" — the caller named a frozen control set. Trusted as stated.
      "identical-output" — the output text hashes match. Proof, not inference.

    We do not guess a third way. An item whose output is missing from the log is
    not an anchor, however tempting it is to assume it didn't change.
    """
    if anchor_ids:
        got = [(a, b) for a, b in pairs if a.item_id in anchor_ids]
        if got:
            return got, "declared"
    got = [(a, b) for a, b in pairs
           if a.output_text and b.output_text and _h(a.output_text) == _h(b.output_text)]
    if got:
        return got, "identical-output"
    return [], "none"


def detect_drift(
    run_a_path: str,
    run_b_path: str,
    anchor_ids: Optional[list] = None,
) -> AuditResult:
    """Compare two judge runs and attribute any score change."""
    a_recs, a_rep = load_judge_records(run_a_path)
    b_recs, b_rep = load_judge_records(run_b_path)
    pairs, warns = pair_runs(a_recs, b_recs)

    res = AuditResult(kind="drift")
    res.notes.extend(a_rep.warnings + b_rep.warnings + warns)

    fp_a, _ = consensus_fingerprint(a_recs)
    fp_b, _ = consensus_fingerprint(b_recs)
    res.fingerprint = fp_b
    res.n_items = len(pairs)

    verdict = fp_a.comparable_to(fp_b)
    diffs = fp_a.diff(fp_b)

    # --- 1. the fingerprint. Mechanical, hence certain. ----------------------
    if verdict == "incomparable":
        lines = [f"{f}: {x!r} → {y!r}" for f, x, y in diffs]
        res.findings.append(Finding(
            id="drift.fingerprint",
            title=f"Judge changed between runs — {len(diffs)} field(s) differ",
            severity=SEV_INVALIDATING,
            confidence=CONF_CERTAIN,
            detail=("These two runs were scored by different instruments. Their means "
                    "are not on the same scale, so the difference between them is not "
                    "an improvement or a regression — it is not a quantity at all "
                    "until the judge's own shift is measured and removed."),
            evidence=lines + [f"A: {fp_a.describe()}", f"B: {fp_b.describe()}"],
            fix=("Re-score run A with run B's judge and compare like with like, or "
                 "keep an anchor set so the judge's shift can be measured directly."),
            why=("A judge score is a joint measurement of the output and the judge. "
                 "Change the judge and you have changed the unit."),
            tags=["drift", "fingerprint"],
        ))
    elif verdict == "unknown":
        res.findings.append(Finding(
            id="drift.no-fingerprint",
            title="No judge fingerprint recorded — comparability is unfalsifiable",
            severity=SEV_SERIOUS,
            confidence=CONF_CERTAIN,
            detail=("Neither run records which model, prompt, rubric, or temperature "
                    "produced its scores. The judge may or may not have changed; "
                    "nothing in these files can settle it, and nothing ever will. "
                    "This is not a gap that can be closed retrospectively."),
            evidence=[f"A: {fp_a.describe()}", f"B: {fp_b.describe()}"],
            fix=("Log judge model, prompt hash, rubric hash, scale and temperature "
                 "with every eval record. It costs one dict and it is the difference "
                 "between an eval you can defend and one you can't."),
            tags=["drift", "fingerprint"],
        ))
    else:
        res.findings.append(Finding(
            id="drift.fingerprint-stable",
            title="Judge fingerprint identical across runs",
            severity=SEV_CLEAN,
            confidence=CONF_CERTAIN,
            detail=f"Both runs judged by: {fp_a.describe()}",
            tags=["drift", "fingerprint"],
        ))

    if not pairs:
        res.findings.append(Finding(
            id="drift.no-overlap",
            title="The two runs share no item_id",
            severity=SEV_INVALIDATING,
            confidence=CONF_CERTAIN,
            detail="Entirely different eval sets. Their means describe different "
                   "populations and the difference between them means nothing.",
            fix="Keep a stable item_id across runs.",
            tags=["drift"],
        ))
        return res

    # --- 2. the headline delta ----------------------------------------------
    sa = [a.score for a, _ in pairs if a.score is not None]
    sb = [b.score for _, b in pairs if b.score is not None]
    if len(sa) != len(pairs) or len(sb) != len(pairs):
        res.notes.append("Some paired records have no numeric score; delta computed "
                         "on the subset that does.")
    if not sa or not sb:
        res.notes.append("No numeric scores — nothing to decompose.")
        return res

    ma, mb = S.mean(sa), S.mean(sb)
    total = mb - ma
    pct = (total / ma * 100.0) if ma else float("nan")

    # --- 3. anchors ----------------------------------------------------------
    anchors, atype = _find_anchors(pairs, set(anchor_ids) if anchor_ids else None)

    if not anchors:
        res.findings.append(Finding(
            id="drift.no-anchors",
            title="No anchor set — the delta cannot be attributed",
            severity=SEV_SERIOUS if verdict != "identical" else SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"Scores moved {ma:.3f} → {mb:.3f} ({pct:+.1f}%) across "
                    f"{len(pairs)} common item(s). How much of that is your system "
                    f"and how much is the judge cannot be determined from these "
                    f"files: no item has a byte-identical output in both runs, and "
                    f"no control set was declared. Splitting the delta would require "
                    f"assuming the judge didn't move, which is the question."),
            effect=total,
            effect_label="delta (unattributable)",
            n=len(pairs),
            fix=("Freeze ~30 items with fixed outputs and re-judge them every run. "
                 "Their score movement is the judge's movement, by construction. "
                 "Then pass them as anchor_ids, or just log output text so they can "
                 "be detected automatically."),
            tags=["drift", "anchors"],
        ))
        return res

    # --- 4. the decomposition ------------------------------------------------
    deltas = [b.score - a.score for a, b in anchors
              if a.score is not None and b.score is not None]
    n_anchor = len(deltas)
    judge_shift = S.mean(deltas)
    ci = S.mean_delta_ci(deltas)
    p, n_pos, n_nz = S.sign_test(deltas)
    d = S.cohens_d_paired(deltas)
    system_shift = total - judge_shift
    share = abs(judge_shift) / abs(total) if abs(total) > 1e-12 else float("inf")

    anchor_base = S.mean([a.score for a, _ in anchors if a.score is not None])
    judge_pct = (judge_shift / anchor_base * 100.0) if anchor_base else float("nan")

    conf = temper_confidence(CONF_LIKELY, n_anchor, POWER_FLOOR_PAIRED)
    if ci[0] == ci[0] and not (ci[0] > 0 or ci[1] < 0):
        conf = temper_confidence("heuristic", n_anchor, POWER_FLOOR_PAIRED)

    anchor_note = ("a declared control set" if atype == "declared"
                   else "items whose output text is byte-identical in both runs")

    if abs(judge_shift) < 1e-9:
        res.findings.append(Finding(
            id="drift.judge-stable",
            title="Judge did not shift on the anchor set",
            severity=SEV_CLEAN,
            confidence=conf,
            detail=(f"Across {n_anchor} anchor item(s) ({anchor_note}) the judge "
                    f"scored identically in both runs. The {pct:+.1f}% headline is "
                    f"your system."),
            effect=0.0,
            effect_label="delta (judge shift)",
            n=n_anchor,
            tags=["drift", "decomposition"],
        ))
        return res

    sev = SEV_CLEAN
    if share >= JUDGE_SHARE_INVALIDATING:
        sev = SEV_INVALIDATING
    elif share >= JUDGE_SHARE_SERIOUS:
        sev = SEV_SERIOUS
    elif share > 0.10:
        sev = SEV_MINOR

    detail = (
        f"Headline: {ma:.3f} → {mb:.3f} ({pct:+.1f}%) over {len(pairs)} common item(s).\n"
        f"   On {n_anchor} anchor item(s) — {anchor_note}, so the system provably did "
        f"not change — the judge scored {judge_shift:+.3f} ({judge_pct:+.1f}%) higher "
        f"in run B.\n"
        f"   That movement is the judge's, by construction. Removing it leaves "
        f"{system_shift:+.3f} for your system.\n"
        f"   The judge accounts for {share:.0%} of the headline."
    )
    if share >= JUDGE_SHARE_INVALIDATING:
        detail += "\n   The improvement is the judge."

    res.findings.append(Finding(
        id="drift.decomposition",
        title=(f"{share:.0%} of the {pct:+.1f}% score change is the judge, not the system"),
        severity=sev,
        confidence=conf,
        detail=detail,
        effect=judge_shift,
        effect_label="delta (judge shift on anchors)",
        ci_low=ci[0],
        ci_high=ci[1],
        n=n_anchor,
        p_value=p,
        evidence=[
            f"total shift      {total:+.3f} ({pct:+.1f}%)",
            f"judge shift      {judge_shift:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
            f"system shift     {system_shift:+.3f}  (total − judge)",
            f"anchor set       {n_anchor} item(s), {atype}",
            f"sign test        {n_pos}/{n_nz} anchors scored higher in B, p={p:.4g}",
            f"paired Cohen's d {d:.2f}" if d == d else "paired Cohen's d n/a",
        ],
        fix=("Re-score both runs with a single pinned judge before quoting any "
             "delta. If the judge must change, report the anchor shift alongside "
             "the headline so readers can subtract it themselves."),
        why=("Anchors hold the system fixed by construction, so their score "
             "movement is attributable to the judge with no modelling assumptions "
             "at all — it is a difference of two measurements of the same object."),
        tags=["drift", "decomposition"],
    ))

    # --- 5. the sign flip: the most embarrassing case ------------------------
    if total * system_shift < 0:
        res.findings.append(Finding(
            id="drift.sign-flip",
            title=f"Direction reverses once judge drift is removed: {pct:+.1f}% → {system_shift:+.3f}",
            severity=SEV_INVALIDATING,
            confidence=conf,
            detail=("The headline and the system move in opposite directions. Your "
                    "system got *worse*; the judge got more generous by more than "
                    "enough to hide it. This is the case where shipping the headline "
                    "means shipping a regression as a win."),
            effect=system_shift,
            effect_label="delta (system, judge-corrected)",
            n=n_anchor,
            fix="Do not report the headline. Re-score under one judge.",
            tags=["drift", "decomposition", "sign-flip"],
        ))

    # --- 6. temperature: even one judge isn't one instrument -----------------
    for tag, fp in (("A", fp_a), ("B", fp_b)):
        if fp.temperature is not None and fp.temperature > 0:
            res.findings.append(Finding(
                id=f"drift.temperature-{tag.lower()}",
                title=f"Run {tag} judged at temperature {fp.temperature} — scores are not repeatable",
                severity=SEV_MINOR,
                confidence=CONF_CERTAIN,
                detail=("A nonzero-temperature judge returns different scores for the "
                        "same input on the same day. Some part of any delta you "
                        "measure is resampling noise, and re-running this eval "
                        "unchanged would move the number. Without repeat runs, that "
                        "part is unquantified."),
                fix="Judge at temperature 0, or average over repeats and report the spread.",
                tags=["drift", "temperature"],
            ))

    if n_anchor < POWER_FLOOR_PAIRED:
        res.findings.append(Finding(
            id="drift.thin-anchors",
            title=f"Only {n_anchor} anchor item(s) — judge shift is loosely estimated",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=(f"The judge-shift interval [{ci[0]:+.3f}, {ci[1]:+.3f}] is wide at "
                    f"this n. The decomposition's direction is likely right; its "
                    f"magnitude is soft, and it is marked heuristic so it will not "
                    f"gate CI."),
            n=n_anchor,
            fix=f"Grow the anchor set to at least {POWER_FLOOR_PAIRED} items.",
            tags=["drift", "power"],
        ))

    return res
