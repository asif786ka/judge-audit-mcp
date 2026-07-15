"""Position bias — does the judge prefer whichever answer it read first?

The manipulation: present the same pair of outputs twice, once as (A, B) and once
as (B, A). The content is identical. A judge with no position bias picks the same
*content* both times. A judge with position bias picks the same *slot* both times
— which, since the contents swapped, is a self-contradiction.

That contradiction is the measurement, and it's clean: there's no "well, maybe it
was right" defence available. The judge said X beats Y and Y beats X.

Two numbers come out, and they answer different questions:

    consistency  — how often the judge agrees with itself. This is a *reliability*
                   number and it caps everything: a judge that contradicts itself
                   on 30% of pairs has a 30% noise floor no rubric change fixes.
    skew         — among the contradictions, which way they lean. A judge that
                   flips randomly is unreliable; one that flips *toward position 1*
                   every time is biased, and biased is worse, because bias doesn't
                   average out across a benchmark the way noise does. It moves
                   your leaderboard in one direction.

Zheng et al. (2023) found GPT-4 as judge flipping on a substantial share of
MT-bench pairs. Everyone cites this. Almost nobody measures it on their own judge.
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

# Contradiction rate above which pairwise verdicts stop supporting a leaderboard.
# 0.25 is a judgement call: at one contradiction in four, a 5-point win between
# two models is inside the judge's own self-disagreement.
INCONSISTENT_SERIOUS = 0.25
INCONSISTENT_INVALIDATING = 0.40


def probe_position(path: str, **kw) -> AuditResult:
    """Expects two records per item_id: variant "" (original) and "swapped".

    Each record's `choice` is the slot the judge picked ("A"/"B"/"tie").
    """
    recs, rep = load_judge_records(path)
    res = AuditResult(kind="bias")
    res.notes.extend(rep.warnings)

    orig = {r.item_id: r for r in recs if r.variant in ("", "original")}
    swap = {r.item_id: r for r in recs if r.variant in ("swapped", "swap", "reversed")}
    ids = sorted(set(orig) & set(swap))

    if not ids:
        res.findings.append(Finding(
            id="bias.position.no-variants",
            title="No swapped variants found — position bias cannot be probed",
            severity=SEV_MINOR,
            confidence=CONF_CERTAIN,
            detail=("This probe needs each comparison judged twice: once as (A,B) and "
                    "once as (B,A), tagged variant='' and variant='swapped' under the "
                    "same item_id."),
            fix=("Re-run the pairwise eval with the two outputs swapped and tag the "
                 "second pass variant='swapped'. Costs one extra judge call per item; "
                 "`emit_probe_set` generates the file for you."),
            tags=["position"],
        ))
        return res

    res.fingerprint = recs[0].fingerprint
    res.n_items = len(ids)

    # In the swapped presentation the slots are reversed, so picking the same
    # *content* means picking the opposite *slot*. This inversion is the crux of
    # the whole probe and is the one place it would be easy to get backwards.
    flip = {"A": "B", "B": "A", "tie": "tie"}
    consistent, contradict, ties = [], [], 0
    first_slot_wins = 0

    for i in ids:
        c1, c2 = orig[i].choice, swap[i].choice
        if c1 is None or c2 is None:
            continue
        c1, c2 = str(c1).strip(), str(c2).strip()
        if c1 == "tie" or c2 == "tie":
            ties += 1
            if c1 == c2:
                consistent.append(i)
            else:
                contradict.append(i)
            continue
        if flip.get(c2) == c1:
            consistent.append(i)      # same content chosen both times
        else:
            contradict.append(i)      # same slot chosen both times → contradiction
            if c1 == "A" and c2 == "A":
                first_slot_wins += 1

    n = len(consistent) + len(contradict)
    if n == 0:
        res.notes.append("No usable choice pairs.")
        return res

    incons = len(contradict) / n
    ci = S.wilson_ci(len(contradict), n)

    sev = SEV_CLEAN
    if incons >= INCONSISTENT_INVALIDATING:
        sev = SEV_INVALIDATING
    elif incons >= INCONSISTENT_SERIOUS:
        sev = SEV_SERIOUS
    elif incons > 0.10:
        sev = SEV_MINOR

    res.findings.append(Finding(
        id="bias.position.consistency",
        title=f"Judge contradicts itself on {incons:.0%} of pairs when order is swapped",
        severity=sev,
        confidence=temper_confidence(CONF_LIKELY, n, POWER_FLOOR_PAIRED),
        detail=(f"{len(contradict)} of {n} comparison(s) got a different winner purely "
                f"because the two answers changed places. The content was identical "
                f"both times, so there is no reading of these where the judge was "
                f"right twice.\n"
                f"   This is a hard floor on the judge's reliability: any margin "
                f"smaller than {incons:.0%} on a pairwise leaderboard is inside the "
                f"judge's own self-disagreement."),
        effect=incons,
        effect_label="contradiction rate",
        ci_low=ci[0],
        ci_high=ci[1],
        n=n,
        evidence=[f"consistent: {len(consistent)}", f"contradictory: {len(contradict)}",
                  f"ties involved: {ties}",
                  f"example contradictions: {', '.join(contradict[:4])}" if contradict else ""],
        fix=("Judge every pair in both orders and keep only the consistent verdicts "
             "(or count contradictions as ties). Doubles judge cost; removes the "
             "artefact entirely."),
        why="Order is not information about quality. Any response to it is bias.",
        citation="Zheng et al. (2023), 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena'",
        tags=["position", "consistency"],
    ))

    # --- direction: noise or a thumb on the scale? --------------------------
    if contradict:
        nc = len(contradict)
        p = S.binom_test_two_sided(first_slot_wins, nc, 0.5)
        rate = first_slot_wins / nc
        sci = S.wilson_ci(first_slot_wins, nc)
        biased = (sci[0] > 0.5 or sci[1] < 0.5)
        conf = temper_confidence(CONF_LIKELY if biased else "heuristic",
                                 nc, POWER_FLOOR_PAIRED)
        direction = "first" if rate > 0.5 else "second"

        res.findings.append(Finding(
            id="bias.position.skew",
            title=(f"Contradictions favour the {direction} position "
                   f"{max(rate, 1 - rate):.0%} of the time"),
            severity=(SEV_SERIOUS if biased and abs(rate - 0.5) > 0.2
                      else SEV_MINOR if biased else SEV_CLEAN),
            confidence=conf,
            detail=("Among the pairs where the judge contradicted itself, it picked "
                    f"the {direction}-presented answer {max(rate, 1 - rate):.0%} of "
                    f"the time (p={p:.3g}).\n"
                    "   This distinction matters more than it looks. Contradictions "
                    "that fall 50/50 are noise: they widen your error bars but they "
                    "cancel across a benchmark. Contradictions that lean one way are "
                    "bias: they do not cancel, they accumulate, and whichever model "
                    "your harness happens to put first wins ties it did not earn."),
            effect=rate,
            effect_label="P(first position wins | contradiction)",
            ci_low=sci[0],
            ci_high=sci[1],
            n=nc,
            p_value=p,
            fix=("Randomise presentation order per item at minimum. Better: judge both "
                 "orders and drop contradictions."),
            citation="Wang et al. (2023), 'Large Language Models are not Fair Evaluators'",
            tags=["position", "skew"],
        ))

    return res
