"""judge-audit-mcp MCP server.

Tools:
  calibrate_judge     - does the judge agree with humans beyond chance? (Cohen's κ)
  detect_judge_drift  - did your system improve, or did the judge change underneath you?
  bias_probe          - position / verbosity / self-preference / length / sycophancy / distribution
  emit_probe_set      - generate the variant file a probe needs you to judge
  audit_judge         - run every probe that this data supports, in one call
  explain_metric      - what κ / anchors / effective levels mean and why they're the right question

Run:  python -m judge_audit_mcp.server        (stdio transport)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .calibrate import calibrate
from .drift import detect_drift
from .emit import emit_probe_set as _emit
from .io import LoadError, load_judge_records
from .models import (
    CONF_CERTAIN,
    CONF_HEURISTIC,
    CONF_LIKELY,
    AuditResult,
    Finding,
    SEV_CLEAN,
    SEV_INVALIDATING,
    SEV_MINOR,
    SEV_SERIOUS,
    severity_at_least,
)
from .probes import FREE_PROBES, PROBES

mcp = FastMCP("judge-audit-mcp")

_ICON = {SEV_INVALIDATING: "🛑", SEV_SERIOUS: "⚠️", SEV_MINOR: "ℹ️", SEV_CLEAN: "✅"}
_CONF_NOTE = {
    CONF_CERTAIN: "certain (mechanical — not an inference)",
    CONF_LIKELY: "likely (interval excludes the null, n clears the power floor)",
    CONF_HEURISTIC: "heuristic (underpowered or interval straddles null — will not gate CI)",
}


def _fmt_finding(f: Finding, verbose: bool = True) -> str:
    lines = [f"{_ICON.get(f.severity, '·')} [{f.severity.upper()}] {f.title}"]
    if f.detail:
        for ln in f.detail.split("\n"):
            lines.append(f"   {ln}" if not ln.startswith("   ") else ln)
    if f.effect is not None:
        lines.append(f"   → {f.effect_label or 'effect'}: {f.describe_effect()}")
    if verbose:
        if f.evidence:
            lines.append("   Evidence:")
            for e in [e for e in f.evidence if e][:6]:
                lines.append(f"     · {e}")
        lines.append(f"   Confidence: {_CONF_NOTE.get(f.confidence, f.confidence)}")
        if f.fix:
            lines.append(f"   Fix: {f.fix}")
        if f.why:
            lines.append(f"   Why: {f.why}")
        if f.citation:
            lines.append(f"   Ref: {f.citation}")
    return "\n".join(lines)


def _fmt_result(res: AuditResult, title: str, verbose: bool = True,
                severity: str = "") -> str:
    shown = [f for f in res.findings
             if not severity or severity_at_least(f.severity, severity)]
    c = res.counts()
    out = [f"# {title}", ""]
    out.append(f"Judge: {res.fingerprint.describe()}")
    if res.n_items:
        out.append(f"Items: {res.n_items}")
    out.append("")
    if not shown:
        out.append("No findings at this severity.")
    else:
        out.append(f"**Verdict: {res.worst().upper()}** — "
                   f"{c[SEV_INVALIDATING]} invalidating, {c[SEV_SERIOUS]} serious, "
                   f"{c[SEV_MINOR]} minor, {c[SEV_CLEAN]} clean.")
        out.append("")
        for f in sorted(shown, key=lambda f: -{"invalidating": 3, "serious": 2,
                                               "minor": 1, "clean": 0}[f.severity]):
            out.append(_fmt_finding(f, verbose))
            out.append("")
    if res.notes and verbose:
        out.append("## Notes")
        for n in res.notes:
            if n:
                out.append(f"   · {n}")
    return "\n".join(out)


def _err(e: Exception) -> str:
    return f"⚠️ {type(e).__name__}: {e}"


# --- tools -------------------------------------------------------------------

@mcp.tool()
def calibrate_judge(
    judge_path: str,
    gold_path: str,
    weights: str = "auto",
    human_ceiling: float = 0.0,
    verbose: bool = True,
) -> str:
    """Measure whether an LLM judge agrees with human labels beyond chance.

    Reports Cohen's κ (weighted, for ordinal scores) with a bootstrap CI, next to
    the two numbers that make it legible: the majority-class baseline, and the
    inter-human ceiling. Accuracy alone is the trap — on a set that's 78% pass, a
    judge that always says "pass" scores 78%.

    Args:
        judge_path: JSONL/CSV of judge outputs. Needs an id and a score/label/choice.
        gold_path: JSONL/CSV of human labels, joined on the same id.
        weights: "auto" | "linear" | "quadratic" | "none". Auto uses weighted κ for
            ordinal scales with >2 levels, which is nearly always what you want —
            plain κ treats 9-vs-8 as exactly as wrong as 9-vs-1.
        human_ceiling: inter-human κ, if you measured it. Without this we cannot
            tell a bad judge from an underdefined task. Also read from a
            `rater_agreement` field in the gold file.
        verbose: include evidence, fixes and citations.
    """
    try:
        res = calibrate(judge_path, gold_path, weights,
                        human_ceiling if human_ceiling > 0 else None)
    except (LoadError, FileNotFoundError, ValueError) as e:
        return _err(e)
    return _fmt_result(res, "Judge calibration", verbose)


@mcp.tool()
def detect_judge_drift(
    run_a_path: str,
    run_b_path: str,
    anchor_ids: str = "",
    verbose: bool = True,
) -> str:
    """Compare two eval runs and attribute the score change to the system or the judge.

    Answers "my scores went up 6% — is that real?". Diffs the judge fingerprint
    (model, prompt hash, rubric hash, scale, temperature) between runs; if it
    changed, the two runs are not on the same scale. Then, using anchor items
    whose outputs are byte-identical across runs — so the system provably did not
    change — measures how much the judge itself moved and subtracts it.

    Without anchors this reports the fingerprint change and refuses to apportion
    the delta, because apportioning it would assume the answer. Freeze ~30 items
    with fixed outputs, re-judge them every run: that's your judge canary.

    Args:
        run_a_path: earlier run (JSONL/CSV).
        run_b_path: later run.
        anchor_ids: comma-separated item_ids of a declared frozen control set.
            Optional — anchors are auto-detected from identical output text.
        verbose: include evidence, fixes and citations.
    """
    ids = [s.strip() for s in anchor_ids.split(",") if s.strip()] if anchor_ids else None
    try:
        res = detect_drift(run_a_path, run_b_path, ids)
    except (LoadError, FileNotFoundError, ValueError) as e:
        return _err(e)
    return _fmt_result(res, "Judge drift", verbose)


@mcp.tool()
def bias_probe(
    path: str,
    probe: str,
    gold_path: str = "",
    judge_family: str = "",
    scale_min: float = 0.0,
    scale_max: float = 0.0,
    claimed_delta: float = 0.0,
    verbose: bool = True,
) -> str:
    """Test an LLM judge for a specific bias.

    Probes:
      position        — swap A/B order; content identical, so any flip is order bias.
                        Needs variant=""/"swapped" records per item_id.
      verbosity       — pad outputs with content-free filler; any lift is bias.
                        Needs variant=""/"padded".
      self_preference — does the judge favour its own family? Measured as residual
                        against a human panel, since raw score gaps are confounded
                        by real quality. Needs output_family + gold_path.
      length_confound — does the judge pay for length among answers humans rated
                        equally? Distinguishes bias from "long answers are better".
                        Needs gold_path + output lengths.
      sycophancy      — does a content-free hint ("this is our new model") move the
                        score? Needs variant=""/"hinted".
      distribution    — no variants needed, runs on any log you already have. Finds
                        granularity collapse, ceiling effects, dead rubric levels,
                        and what your delta means in items-that-actually-moved.

    Args:
        path: judge outputs, including probe variants where the probe needs them.
        probe: one of the above.
        gold_path: human labels (self_preference and length_confound require these).
        judge_family: e.g. "claude" — inferred from the judge model string if logged.
        scale_min/scale_max: declared rubric bounds for `distribution`. Pass these:
            inferring the scale from the judge's own output would define the
            "never uses the bottom half" pathology out of existence.
        claimed_delta: for `distribution` — a headline delta to translate into
            "how many items actually moved a notch".
        verbose: include evidence, fixes and citations.
    """
    fn = PROBES.get(probe)
    if not fn:
        return (f"Unknown probe {probe!r}. Available: {', '.join(sorted(PROBES))}. "
                f"Probes needing no extra judge calls: {', '.join(FREE_PROBES)}.")
    kw: dict = {"gold_path": gold_path, "judge_family": judge_family}
    if probe == "distribution":
        kw = {
            "scale_min": scale_min if scale_min or scale_max else None,
            "scale_max": scale_max if scale_min or scale_max else None,
            "claimed_delta": claimed_delta or None,
        }
    try:
        res = fn(path, **kw)
    except (LoadError, FileNotFoundError, ValueError) as e:
        return _err(e)
    return _fmt_result(res, f"Bias probe — {probe}", verbose)


@mcp.tool()
def audit_judge(
    path: str,
    gold_path: str = "",
    judge_family: str = "",
    scale_min: float = 0.0,
    scale_max: float = 0.0,
    verbose: bool = False,
) -> str:
    """Run every check this data supports, and say what's missing for the rest.

    The "I have an eval log, is my judge okay?" entry point. Runs the free probes
    unconditionally, adds calibration and the human-controlled probes if gold
    labels are supplied, and reports which probes need variants you haven't
    generated yet — with the command to generate them.

    Args:
        path: judge outputs.
        gold_path: human labels. Without these, roughly half the checks are
            unavailable, and the ones that need a quality control will say so
            rather than reporting a confounded number.
        judge_family: e.g. "claude".
        scale_min/scale_max: declared rubric bounds.
        verbose: full evidence for every finding (long).
    """
    try:
        recs, rep = load_judge_records(path)
    except (LoadError, FileNotFoundError) as e:
        return _err(e)

    out = [f"# Judge audit — {Path(path).name}", ""]
    out.append(rep.describe())
    out.append("")

    sections: list[tuple[str, AuditResult]] = []
    for name in ("distribution", "length_confound", "position", "verbosity",
                 "sycophancy", "self_preference"):
        kw: dict = {"gold_path": gold_path, "judge_family": judge_family}
        if name == "distribution":
            kw = {"scale_min": scale_min or None, "scale_max": scale_max or None,
                  "claimed_delta": None}
        try:
            sections.append((name, PROBES[name](path, **kw)))
        except (LoadError, ValueError) as e:
            out.append(f"· {name}: skipped — {e}")

    if gold_path:
        try:
            sections.append(("calibration", calibrate(path, gold_path)))
        except (LoadError, FileNotFoundError, ValueError) as e:
            out.append(f"· calibration: skipped — {e}")
    else:
        out.append("· calibration: skipped — no gold_path. Without human labels there "
                   "is no way to know whether this judge agrees with anyone.")
        out.append("")

    worst = SEV_CLEAN
    for _, res in sections:
        w = res.worst()
        if {"invalidating": 3, "serious": 2, "minor": 1, "clean": 0}[w] > \
           {"invalidating": 3, "serious": 2, "minor": 1, "clean": 0}[worst]:
            worst = w

    out.append(f"**Overall: {worst.upper()}**")
    out.append("")
    for name, res in sections:
        gating = [f for f in res.findings
                  if severity_at_least(f.severity, SEV_MINOR)]
        if not gating:
            out.append(f"## {name} — ✅ nothing found")
            out.append("")
            continue
        out.append(f"## {name}")
        out.append("")
        for f in sorted(gating, key=lambda f: -{"invalidating": 3, "serious": 2,
                                                "minor": 1, "clean": 0}[f.severity]):
            out.append(_fmt_finding(f, verbose))
            out.append("")
    return "\n".join(out)


@mcp.tool()
def emit_probe_set(path: str, probe: str, out_path: str = "",
                   hint: str = "new_model") -> str:
    """Generate the variant file a probe needs your judge to score.

    Writes a JSONL of items to judge (swapped orders / padded outputs / hinted
    prompts), each tagged with `variant` and keeping its original item_id. Run your
    own judge over it, fill in score/choice, then pass the result to bias_probe.

    This round trip is why the core of this server needs no API key and costs
    nothing to run.

    Args:
        path: an existing judge run to build variants from.
        probe: "position" | "verbosity" | "sycophancy". The others need no variants —
            distribution and length_confound run on your existing log directly.
        out_path: defaults to <path>.<probe>.jsonl
        hint: for sycophancy — new_model | production | team_favourite | expensive.
    """
    try:
        dest, n = _emit(path, probe, out_path, hint)
    except (LoadError, FileNotFoundError, ValueError) as e:
        return _err(e)
    return (f"Wrote {n} record(s) to {dest}\n\n"
            f"Next: score these with your judge, fill in the score/choice field, then:\n"
            f"    bias_probe(path='{dest}', probe='{probe}')\n\n"
            f"Keep item_id and variant untouched — the probe pairs on them.")


@mcp.tool()
def explain_metric(metric: str) -> str:
    """Explain what a judge-audit metric means and why it's the right question.

    Args:
        metric: kappa | anchors | fingerprint | effective_levels | position_bias |
            self_preference | length_confound | sycophancy | power | ceiling
    """
    m = (metric or "").lower().strip().replace("-", "_").replace(" ", "_")
    docs = {
        "kappa": (
            "**Cohen's κ** — agreement corrected for chance.\n\n"
            "    κ = (po − pe) / (1 − pe)\n\n"
            "po is how often judge and human agree; pe is how often they'd agree by "
            "luck given their base rates. κ = 0 means 'no better than guessing at the "
            "right rate'; κ = 1 means perfect.\n\n"
            "Why not accuracy: on a benchmark where 78% of items pass, a judge that "
            "says 'pass' unconditionally is 78% accurate. That number looks fine in a "
            "slide and means nothing. κ = 0 for that judge, which is the truth.\n\n"
            "Use *weighted* κ for ordinal scores. Plain κ treats a judge that said 9 "
            "where the human said 8 as exactly as wrong as one that said 1.\n\n"
            "The catch nobody mentions: κ is bounded by your labels' own reliability. "
            "If your annotators agree with each other at κ=0.55, a judge at κ=0.50 is "
            "at the ceiling and cannot be improved — the task is underdefined, not the "
            "judge. Always measure inter-human agreement before blaming the judge."
        ),
        "anchors": (
            "**Anchor items** — items whose output is byte-identical across two runs.\n\n"
            "The system provably did not change for these, so any movement in their "
            "scores is the judge moving. Not an assumption — an identity. It's the "
            "difference between two measurements of the same object.\n\n"
            "    judge_shift  = mean(score_B − score_A) over anchors\n"
            "    system_shift = total_shift − judge_shift\n\n"
            "Without anchors, a score delta cannot be attributed: mean(B) − mean(A) "
            "confounds the system changing, the eval set changing, and the judge "
            "changing, and no statistic separates them after the fact.\n\n"
            "The fix costs about a dollar: **freeze ~30 items with fixed outputs and "
            "re-judge them every run.** That's a judge canary. Almost nobody has one, "
            "which is why almost nobody catches judge drift."
        ),
        "fingerprint": (
            "**Judge fingerprint** — model + prompt hash + rubric hash + scale + "
            "temperature + provider.\n\n"
            "Change any of these and scores from before and after are not on the same "
            "scale. Not 'roughly comparable' — not comparable.\n\n"
            "Every field is something people change without thinking of it as changing "
            "the measuring instrument: 'we upgraded to the new Sonnet', a PR titled "
            "'clarify judge prompt', a temperature nudge. None of it appears in the "
            "changelog of the system under test, because the judge isn't the system "
            "under test. That's exactly why it goes unnoticed.\n\n"
            "The most common finding is the worst one: no fingerprint recorded at all. "
            "That's not evidence the judge stayed the same — it's the permanent "
            "impossibility of ever showing that it did. Log it now; it cannot be "
            "recovered later."
        ),
        "effective_levels": (
            "**Effective levels** — 2^entropy of the judge's score distribution: how "
            "many levels the judge *behaves* as if it has, versus how many the rubric "
            "claims.\n\n"
            "Your rubric says 1-10. Your judge emits 7, 8, and sometimes 9. Effective "
            "resolution ≈ 2. Everyone reading '7.6 out of 10' believes they're looking "
            "at ten rungs of measurement. There are two.\n\n"
            "This matters most for deltas. If the judge emits integers, a +0.43 mean "
            "change over 120 items isn't 120 items each improving a bit — it's ~52 "
            "items hopping a single threshold while 68 didn't move at all. Whether "
            "that's a result depends on whether those 52 were sitting on a boundary "
            "the judge is coin-flipping on. Quoting the mean to two decimals means "
            "nobody ever asks.\n\n"
            "Costs nothing to check. It's in the log you already have."
        ),
        "position_bias": (
            "**Position bias** — does the judge prefer whichever answer it read first?\n\n"
            "Present the same pair twice, (A,B) and (B,A). A judge with no position "
            "bias picks the same content both times. One with position bias picks the "
            "same slot both times — a flat self-contradiction, since the contents "
            "swapped.\n\n"
            "Two numbers, different meanings. *Consistency* is reliability: contradict "
            "yourself on 30% of pairs and that's a 30% noise floor on every pairwise "
            "claim you make. *Skew* is bias: contradictions that fall 50/50 are noise "
            "and cancel across a benchmark; contradictions that lean toward position 1 "
            "accumulate, and whichever model your harness happens to put first wins "
            "ties it didn't earn.\n\n"
            "Ref: Zheng et al. (2023); Wang et al. (2023)."
        ),
        "self_preference": (
            "**Self-preference** — does the judge favour its own family's outputs?\n\n"
            "The naive test — compare the judge's mean for Claude outputs vs GPT "
            "outputs — is invalid, and confidently so. Maybe the Claude outputs were "
            "better. Judge scores alone cannot separate 'favours its own' from 'its own "
            "was better'. That's a confound, not a sample-size problem; more data never "
            "fixes it.\n\n"
            "So measure the *residual against a human panel*:\n\n"
            "    residual(family) = mean(judge_score − human_score)\n\n"
            "Humans price in real quality. What's left is the judge's disagreement with "
            "them, per family. If the judge over-scores its own by +0.8 relative to "
            "humans and others by −0.1, that 0.9 gap isn't quality — humans already "
            "accounted for quality.\n\n"
            "Where it bites: model bakeoffs. The judge is usually whatever's best and "
            "cheapest, which is usually the same family as one of the candidates.\n\n"
            "Ref: Panickssery et al. (2024)."
        ),
        "length_confound": (
            "**Length confound** — is the judge rewarding length, or is length just "
            "correlated with quality?\n\n"
            "'Longer answers really are better, so a judge scoring them higher is "
            "right' is a fair objection to any correlation you show. Length and quality "
            "*are* correlated in real outputs.\n\n"
            "Control for quality with the human score:\n\n"
            "    judge_score ~ b0 + b1·human_score + b2·length\n\n"
            "b2 is what the judge pays for length *among answers humans rated equally*. "
            "If length were merely a proxy for quality, b2 ≈ 0 — once you know how good "
            "an answer is, its length adds nothing. b2 > 0 means the judge pays for "
            "length on top of quality, and the defence collapses.\n\n"
            "Pair it with the `verbosity` probe: that one is interventional (padding is "
            "quality-preserving by construction, clean identification) and this one is "
            "observational on real data with the confound controlled. Together they're "
            "much stronger than either alone.\n\n"
            "Ref: Dubois et al. (2024), Length-Controlled AlpacaEval."
        ),
        "sycophancy": (
            "**Sycophancy** — does the judge move when told what you're hoping for?\n\n"
            "Re-judge identical outputs with a content-free hint: 'this is from our new "
            "model'. The hint says something about the asker, nothing about the answer. "
            "Any response to it is the judge scoring your hopes.\n\n"
            "Why this one is embarrassing: nobody writes 'please favour our new model' "
            "on purpose. Hints leak in. The judge prompt says 'compare the baseline "
            "response to the improved response' and the variable is named "
            "`improved_response`. The rubric names the model under test. Every one of "
            "those is a hint, none was intended, and all of them are in production "
            "somewhere right now.\n\n"
            "Existence matters more than size here: whoever writes the prompt controls "
            "that channel, and they're the person with a stake in the outcome.\n\n"
            "Ref: Sharma et al. (2023)."
        ),
        "power": (
            "**Power** — why findings here get downgraded at small n.\n\n"
            "The bias literature reports effects that are real but modest: position "
            "flips in the 10-30% range, self-preference well under a point on a 10-point "
            "scale. Effects that size are trivially manufactured by sampling noise at "
            "n=20 plus a hopeful analyst.\n\n"
            "So: below 30 paired observations (50 for κ, whose variance blows up when "
            "one class dominates — exactly the regime real eval sets live in), no "
            "statistical finding may claim better than 'heuristic', whatever its "
            "p-value, and heuristic findings never fail a build.\n\n"
            "A tool that red-builds on an underpowered estimate gets switched off within "
            "a week, and then it audits nothing at all. The exception is mechanical "
            "findings — 'the judge model string differs between these two runs' is a "
            "string comparison, not an inference, and needs no sample at all."
        ),
        "ceiling": (
            "**Ceiling effect** — most of your items already score maximum.\n\n"
            "If 60% of items are at the top of the scale, improvements to those items "
            "cannot show up. There's nowhere up to go. Your eval will report 'no "
            "significant change' indefinitely while the system genuinely improves on "
            "60% of its inputs, and only the remaining 40% can move at all, so any "
            "effect you do measure is diluted by roughly that factor.\n\n"
            "The mirror image is a floor effect, where regressions are invisible and "
            "the eval understates harm.\n\n"
            "Fix: harden the eval set until the distribution has room above it."
        ),
    }
    if m in docs:
        return docs[m]
    return (f"Unknown metric {metric!r}. Available: {', '.join(sorted(docs))}.")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
