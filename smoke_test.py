"""End-to-end smoke test for judge-audit-mcp.

    python smoke_test.py

Runs the actual MCP tool functions, the engines and the CLI against the fixture
sets, and asserts the verdicts are right. No network, no API keys.

Three kinds of assertion here, and the second and third are the ones that matter:

1.  **The tool fires when it should.** Easy, and not very convincing on its own —
    any function that always returns "BIASED" passes these.

2.  **The tool stays quiet when it should.** Every scenario has a control fixture
    built by the same generator with the bias mechanism removed: a drift pair
    with the judge held fixed, a length-biased judge next to an innocent one with
    the same naive correlation, a consistent judge, a healthy score distribution.
    A bias detector that cannot be made to shut up is a random number generator,
    and these are the tests that prove this one can.

3.  **The statistics are right.** Every estimator in `stats.py` is cross-checked
    against scipy/scikit-learn where available, and skipped where they aren't.
    Reimplementing Cohen's κ to avoid an 80MB dependency is only defensible if
    the reimplementation is demonstrably correct — so `pip install scipy
    scikit-learn` and re-run to check that claim rather than taking it.

The fixtures are generated, not hand-written (`python fixtures/generate.py`), so
these numbers come out of a stated mechanism rather than out of someone tuning a
JSON file until the test went green.
"""
from __future__ import annotations

import io as _io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

PASS, FAIL = 0, 0
ROOT = Path(__file__).parent
FIX = ROOT / "fixtures"


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}\n      {str(detail)[:400]}")


def section(name: str) -> None:
    print(f"\n{name}")
    print("─" * min(len(name), 70))


sys.path.insert(0, str(ROOT))

from judge_audit_mcp import server as srv                     # noqa: E402
from judge_audit_mcp import stats as S                        # noqa: E402
from judge_audit_mcp.calibrate import calibrate               # noqa: E402
from judge_audit_mcp.drift import detect_drift                # noqa: E402
from judge_audit_mcp.emit import emit_probe_set               # noqa: E402
from judge_audit_mcp.io import (                              # noqa: E402
    LoadError, load_judge_records, pair_runs,
)
from judge_audit_mcp.models import (                          # noqa: E402
    CONF_CERTAIN, CONF_HEURISTIC, JudgeFingerprint,
    SEV_CLEAN, SEV_INVALIDATING, SEV_MINOR, SEV_SERIOUS,
    temper_confidence,
)
from judge_audit_mcp.probes import PROBES                     # noqa: E402


def find(res, fid):
    return next((f for f in res.findings if f.id == fid), None)


# =============================================================================
print("\njudge-audit-mcp — smoke test")
print("=" * 72)

# =============================================================================
section("1. JudgeFingerprint — the load-bearing abstraction")

a = JudgeFingerprint.from_config(model="claude-3-5-sonnet", prompt="Grade 1-10.",
                                 rubric="10=great", scale="1-10", temperature=0.0)
b = JudgeFingerprint.from_config(model="claude-sonnet-5", prompt="Grade 1-10.",
                                 rubric="10=great", scale="1-10", temperature=0.0)
c = JudgeFingerprint.from_config(model="claude-3-5-sonnet", prompt="Grade 1-10.",
                                 rubric="10=great", scale="1-10", temperature=0.0)
# Same prompt, reflowed. A formatter run is not a rubric change.
d = JudgeFingerprint.from_config(model="claude-3-5-sonnet", prompt="Grade\n  1-10.",
                                 rubric="10=great", scale="1-10", temperature=0.0)

check("identical configs are comparable", a.comparable_to(c) == "identical")
check("model bump makes runs incomparable", a.comparable_to(b) == "incomparable")
check("whitespace-only prompt change is NOT a change",
      a.comparable_to(d) == "identical", a.diff(d))
check("empty fingerprint is 'unknown', not 'identical'",
      JudgeFingerprint().comparable_to(a) == "unknown")
check("empty vs empty is still 'unknown' (absence ≠ evidence)",
      JudgeFingerprint().comparable_to(JudgeFingerprint()) == "unknown")
check("prompt text is hashed, never stored",
      "Grade" not in a.prompt_hash and len(a.prompt_hash) == 12)
check("a temperature nudge alone breaks comparability",
      a.comparable_to(JudgeFingerprint.from_config(
          model="claude-3-5-sonnet", prompt="Grade 1-10.", rubric="10=great",
          scale="1-10", temperature=0.7)) == "incomparable")
check("rubric edit alone breaks comparability",
      a.comparable_to(JudgeFingerprint.from_config(
          model="claude-3-5-sonnet", prompt="Grade 1-10.", rubric="10=perfect",
          scale="1-10", temperature=0.0)) == "incomparable")
check("diff names the field that moved",
      a.diff(b)[0][0] == "model", a.diff(b))

# =============================================================================
section("2. Statistics — cross-checked against scipy/sklearn where installed")

try:
    import numpy as np
    from scipy import stats as sp
    from sklearn.metrics import cohen_kappa_score
    HAVE_REF = True
except ImportError:
    HAVE_REF = False
    print("  · scipy/scikit-learn not installed — cross-checks skipped.")
    print("    `pip install judge-audit-mcp[verify]` and re-run to verify the")
    print("    dependency-free implementations against the references.")

# Sample data is built unconditionally. It was briefly built inside the
# `if HAVE_REF:` block, which meant the whole suite crashed on a machine without
# scipy — i.e. exactly the configuration this package claims to support, and the
# one every user installs by default. Left as a comment because it is a small,
# funny example of the thing this server is about: the test that only passes when
# the optional thing is present tells you nothing about the case you ship.
import random                                                    # noqa: E402
rng = random.Random(7)

x = [1 if rng.random() < 0.8 else 0 for _ in range(200)]
y = [v if rng.random() < 0.8 else 1 - v for v in x]
u = [rng.gauss(0, 1) for _ in range(100)]
v = [t * 0.6 + rng.gauss(0, 1) for t in u]
X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
yy = [2.0 + 1.5 * r[0] - 0.7 * r[1] + rng.gauss(0, 0.3) for r in X]
vals = [rng.choice([7, 8, 8, 8, 9]) for _ in range(300)]
ord_pairs = {}
for w in ("linear", "quadratic"):
    p = [rng.randint(1, 10) for _ in range(300)]
    q = [max(1, min(10, t + rng.choice([-1, 0, 0, 1]))) for t in p]
    ord_pairs[w] = (p, q)

if HAVE_REF:
    check("Cohen's κ == sklearn.cohen_kappa_score",
          abs(S.cohens_kappa(x, y) - cohen_kappa_score(x, y)) < 1e-9)

    for w, (p, q) in ord_pairs.items():
        check(f"weighted κ ({w}) == sklearn",
              abs(S.weighted_kappa(p, q, w) - cohen_kappa_score(p, q, weights=w)) < 1e-9)

    for k, n in ((12, 20), (3, 20), (0, 8), (45, 50)):
        check(f"exact binomial test k={k} n={n} == scipy.binomtest",
              abs(S.binom_test_two_sided(k, n) - sp.binomtest(k, n, 0.5).pvalue) < 1e-12)

    lo, hi = S.wilson_ci(12, 20)
    rlo, rhi = sp.binomtest(12, 20).proportion_ci(method="wilson")
    check("Wilson interval == scipy proportion_ci",
          abs(lo - rlo) < 1e-6 and abs(hi - rhi) < 1e-6)

    check("Pearson r == scipy.pearsonr",
          abs(S.pearson_r(u, v) - sp.pearsonr(u, v)[0]) < 1e-10)

    beta = S.ols(X, yy)
    D = np.hstack([np.ones((200, 1)), np.array(X)])
    ref = np.linalg.lstsq(D, np.array(yy), rcond=None)[0]
    check("OLS coefficients == numpy.lstsq",
          all(abs(beta[i] - ref[i]) < 1e-8 for i in range(3)))

    cnt = np.unique(vals, return_counts=True)[1]
    check("entropy == scipy.entropy(base=2)",
          abs(S.entropy_bits(vals) - sp.entropy(cnt / cnt.sum(), base=2)) < 1e-10)

# Reference-free sanity checks on the same estimators, so the suite still tests
# something real when scipy is absent.
check("weighted κ ≥ plain κ for near-miss ordinal ratings",
      S.weighted_kappa(*ord_pairs["linear"], "linear") >
      S.cohens_kappa(*ord_pairs["linear"]))
check("quadratic κ ≥ linear κ (near misses penalised less)",
      S.weighted_kappa(*ord_pairs["quadratic"], "quadratic") >=
      S.weighted_kappa(*ord_pairs["quadratic"], "linear"))
check("κ = 1 for a rater agreeing with itself", abs(S.cohens_kappa(x, x) - 1.0) < 1e-9)
check("OLS recovers planted coefficients (2.0, 1.5, -0.7)",
      all(abs(a - b) < 0.1 for a, b in zip(S.ols(X, yy), (2.0, 1.5, -0.7))),
      S.ols(X, yy))
check("Wilson interval brackets the point estimate",
      S.wilson_ci(12, 20)[0] < 0.6 < S.wilson_ci(12, 20)[1])
check("binomial p=1.0 at the null", abs(S.binom_test_two_sided(10, 20) - 1.0) < 1e-12)

# Properties that hold with or without the reference libraries.
check("κ = 0 for a judge that ignores the input and always says 'pass'",
      abs(S.cohens_kappa(["pass"] * 100,
                         ["pass"] * 78 + ["fail"] * 22)) < 1e-9)
check("majority baseline recovers the base rate",
      abs(S.majority_baseline(["pass"] * 78 + ["fail"] * 22)[0] - 0.78) < 1e-9)
check("effective_levels ≈ 1 for a constant judge",
      abs(S.effective_levels([8] * 100) - 1.0) < 1e-9)
check("effective_levels ≈ n for a uniform judge",
      abs(S.effective_levels(list(range(1, 11)) * 30) - 10.0) < 1e-6)
check("bootstrap is deterministic across calls",
      S.bootstrap_ci(lambda i: S.mean([u[j] for j in i]), len(u)) ==
      S.bootstrap_ci(lambda i: S.mean([u[j] for j in i]), len(u)))
check("bootstrap_ci returns nan for n<2 rather than inventing an interval",
      all(v != v for v in S.bootstrap_ci(lambda i: 1.0, 1)))
check("ols returns None rather than fitting n<=k points",
      S.ols([[1.0, 2.0]], [1.0]) is None)
check("pearson_r is nan (not 0) when a variable has zero variance",
      S.pearson_r([1, 1, 1, 1], [1, 2, 3, 4]) != S.pearson_r([1, 1, 1, 1], [1, 2, 3, 4]))

# =============================================================================
section("3. Power — findings downgrade themselves when n is too small")

check("likely → heuristic below the power floor",
      temper_confidence("likely", 11, 30) == CONF_HEURISTIC)
check("likely survives above the power floor",
      temper_confidence("likely", 60, 30) == "likely")
check("certain is never downgraded (a string compare needs no sample)",
      temper_confidence(CONF_CERTAIN, 2, 50) == CONF_CERTAIN)

# =============================================================================
section("4. Drift — 'the 6% that wasn't'")

res = detect_drift(str(FIX / "drift/run_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"))
fp = find(res, "drift.fingerprint")
dec = find(res, "drift.decomposition")

check("judge model change is caught", fp is not None and fp.severity == SEV_INVALIDATING)
check("fingerprint finding is CERTAIN, not inferred", fp.confidence == CONF_CERTAIN)
check("only the model field is flagged (prompt/rubric/temp unchanged)",
      sum(1 for e in fp.evidence if "→" in e) == 1, fp.evidence)
check("decomposition ran", dec is not None)
check("headline is ~+6%", "+6.0%" in dec.detail, dec.detail[:120])
check("judge is blamed for >80% of the headline", dec.effect / 0.414 > 0.8,
      f"judge shift {dec.effect}")
check("verdict is INVALIDATING", dec.severity == SEV_INVALIDATING)
check("the phrase that sells it", "The improvement is the judge." in dec.detail)
check("judge-shift CI excludes zero", dec.ci_low > 0, (dec.ci_low, dec.ci_high))
check("40 anchors auto-detected from identical output text", dec.n == 40, dec.n)
check("sign test reported", dec.p_value is not None and dec.p_value < 0.05)

# The same thing via declared anchors rather than text hashing.
ids = (FIX / "drift/anchors.txt").read_text().strip()
res_d = detect_drift(str(FIX / "drift/run_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"),
                     ids.split(","))
dec_d = find(res_d, "drift.decomposition")
check("declared anchors give the same answer as detected ones",
      abs(dec_d.effect - dec.effect) < 1e-9)

# --- the control: must NOT cry wolf --------------------------------------
res = detect_drift(str(FIX / "drift/run_clean_v1.jsonl"),
                   str(FIX / "drift/run_clean_v2.jsonl"))
check("CONTROL: same judge, real improvement → no findings above clean",
      res.worst() == SEV_CLEAN, [f.title for f in res.findings])
check("CONTROL: fingerprint reported stable",
      find(res, "drift.fingerprint-stable") is not None)
check("CONTROL: judge shift on anchors is exactly zero",
      find(res, "drift.judge-stable") is not None)
check("CONTROL: nothing gates CI at 'minor'", not res.gating(SEV_MINOR),
      [f.title for f in res.gating(SEV_MINOR)])

# --- no fingerprint: the commonest real-world state -----------------------
res = detect_drift(str(FIX / "drift/run_v1_nofingerprint.jsonl"),
                   str(FIX / "drift/run_v2_nofingerprint.jsonl"))
nf = find(res, "drift.no-fingerprint")
check("missing fingerprint is itself a SERIOUS finding",
      nf is not None and nf.severity == SEV_SERIOUS)
check("missing fingerprint says it can't be fixed retrospectively",
      "retrospectively" in nf.detail)
check("anchors still work without a fingerprint",
      find(res, "drift.decomposition") is not None)

# =============================================================================
section("5. Calibration — 'the judge that flips a coin'")

res = calibrate(str(FIX / "calibration/judge.jsonl"),
                str(FIX / "calibration/human_labels.jsonl"))
k = find(res, "cal.kappa")
bl = find(res, "cal.baseline")

check("κ computed", k is not None)
check("κ lands in the 'fair' band (0.21-0.40)", 0.21 <= k.effect < 0.41, k.effect)
check("accuracy is >80% — the flattering number", "83.0%" in k.detail, k.detail[:90])
check("baseline is surfaced next to it", bl is not None)
check("the constant-'pass' baseline is ~78%", "78.5%" in bl.detail, bl.detail[:120])
check("verdict is SERIOUS", k.severity == SEV_SERIOUS)
check("κ has a bootstrap CI", k.ci_low is not None and k.ci_low < k.effect < k.ci_high)
check("n=200 clears the κ power floor → 'likely'", k.confidence == "likely")
check("human ceiling (κ=0.73) read from the gold file",
      find(res, "cal.below-ceiling") is not None)
check("judge is correctly blamed, not the task",
      "belongs to the judge" in find(res, "cal.below-ceiling").detail)

# --- a good judge on the same items ---------------------------------------
res = calibrate(str(FIX / "calibration/judge_good.jsonl"),
                str(FIX / "calibration/human_labels.jsonl"))
kg = find(res, "cal.kappa")
check("CONTROL: well-calibrated judge scores κ > 0.6", kg.effect > 0.6, kg.effect)
check("CONTROL: no baseline complaint for a good judge",
      find(res, "cal.baseline") is None)

# --- the ceiling case: judge is mediocre, humans are too -------------------
res = calibrate(str(FIX / "calibration/judge.jsonl"),
                str(FIX / "calibration/human_labels_noisy.jsonl"))
ceil = find(res, "cal.ceiling")
check("CEILING: same mediocre judge + noisy humans → 'at the human ceiling'",
      ceil is not None, [f.id for f in res.findings])
check("CEILING: correctly blames the task, not the judge",
      ceil is not None and "the task is" in ceil.detail)
check("CEILING: severity is CLEAN — nothing to fix in the judge",
      ceil is not None and ceil.severity == SEV_CLEAN)

# --- explicit ceiling overrides the file ----------------------------------
res = calibrate(str(FIX / "calibration/judge.jsonl"),
                str(FIX / "calibration/human_labels.jsonl"), human_ceiling=0.40)
check("explicit human_ceiling arg is honoured",
      find(res, "cal.ceiling") is not None)

# =============================================================================
section("6. Bias probes — each must fire on the pathological fixture...")

res = PROBES["position"](str(FIX / "probes/position.jsonl"))
cons, skew = find(res, "bias.position.consistency"), find(res, "bias.position.skew")
check("position: contradiction rate ~26%", 0.20 < cons.effect < 0.32, cons.effect)
check("position: verdict SERIOUS", cons.severity == SEV_SERIOUS)
check("position: skew toward first slot ~80%", skew.effect > 0.7, skew.effect)
check("position: skew CI excludes 0.5 (bias, not noise)", skew.ci_low > 0.5,
      (skew.ci_low, skew.ci_high))
check("position: distinguishes noise from skew in the text",
      "cancel" in skew.detail and "accumulate" in skew.detail)

res = PROBES["verbosity"](str(FIX / "probes/verbosity.jsonl"))
v = find(res, "bias.verbosity.lift")
check("verbosity: padding buys ~0.7 points", 0.5 < v.effect < 0.9, v.effect)
check("verbosity: CI excludes zero", v.ci_low > 0)
check("verbosity: verdict SERIOUS", v.severity == SEV_SERIOUS)

res = PROBES["sycophancy"](str(FIX / "probes/sycophancy.jsonl"))
sy = find(res, "bias.sycophancy.shift")
check("sycophancy: a content-free hint moves the judge ~0.5", sy.effect > 0.3, sy.effect)
check("sycophancy: CI excludes zero", sy.ci_low > 0)
check("sycophancy: warns about accidental hints in real prompts",
      "improved_response" in sy.detail)

res = PROBES["self_preference"](str(FIX / "self-preference/judge.jsonl"),
                                gold_path=str(FIX / "self-preference/human_labels.jsonl"))
sp_f = find(res, "bias.self.delta")
check("self-pref: recovers the ~0.9 planted delta", 0.75 < sp_f.effect < 1.05, sp_f.effect)
check("self-pref: CI excludes zero", sp_f.ci_low > 0)
check("self-pref: verdict INVALIDATING", sp_f.severity == SEV_INVALIDATING)
check("self-pref: residual is measured against humans, not raw scores",
      "priced in" in sp_f.detail)

res = PROBES["distribution"](str(FIX / "probes/distribution_collapsed.jsonl"),
                             scale_min=1, scale_max=10, claimed_delta=0.43)
g = find(res, "bias.dist.granularity")
dead = find(res, "bias.dist.dead-levels")
quant = find(res, "bias.dist.quantisation")
check("distribution: effective resolution ~2.5 on a 1-10 scale",
      2.0 < g.effect < 3.0, g.effect)
check("distribution: granularity finding is CERTAIN (it's a histogram)",
      g.confidence == CONF_CERTAIN)
check("distribution: 7 dead rubric levels found", dead.effect == 7, dead.effect)
check("distribution: +0.43 delta translated into ~129 items moving one notch",
      125 < quant.effect < 133, quant.effect)
check("distribution: declared scale used, not inferred from the judge's own output",
      "1-10" in g.detail or "10 levels" in g.detail)

res = PROBES["length_confound"](str(FIX / "probes/length_biased_judge.jsonl"),
                                gold_path=str(FIX / "probes/length_biased_human.jsonl"))
lb = find(res, "bias.length.controlled")
check("length: biased judge shows a positive controlled slope", lb.effect > 0.5, lb.effect)
check("length: controlled CI excludes zero", lb.ci_low > 0, (lb.ci_low, lb.ci_high))
check("length: rebuts the 'longer really is better' defence",
      "does not hold" in lb.detail)

# =============================================================================
section("7. ...and must stay QUIET on the control fixture")

res = PROBES["position"](str(FIX / "probes/position_clean.jsonl"))
cons = find(res, "bias.position.consistency")
check("CONTROL: consistent judge → ~3% contradictions, CLEAN",
      cons.effect < 0.08 and cons.severity == SEV_CLEAN, cons.effect)
check("CONTROL: skew from 4 contradictions is heuristic, cannot gate",
      find(res, "bias.position.skew").confidence == CONF_HEURISTIC)
check("CONTROL: nothing gates CI", not res.gating(SEV_MINOR))

res = PROBES["distribution"](str(FIX / "probes/distribution_healthy.jsonl"),
                             scale_min=1, scale_max=10)
check("CONTROL: healthy judge uses ~8 effective levels → CLEAN",
      find(res, "bias.dist.granularity").severity == SEV_CLEAN)
check("CONTROL: no ceiling complaint on a healthy distribution",
      find(res, "bias.dist.ceiling") is None)
check("CONTROL: no dead-level complaint", find(res, "bias.dist.dead-levels") is None)

# The headline control: same naive correlation, opposite verdict.
res = PROBES["length_confound"](str(FIX / "probes/length_innocent_judge.jsonl"),
                                gold_path=str(FIX / "probes/length_innocent_human.jsonl"))
li = find(res, "bias.length.controlled")
check("CONTROL: innocent judge's naive r is just as high (~0.78)",
      any("+0.77" in e or "+0.78" in e for e in li.evidence), li.evidence[:1])
check("CONTROL: but its controlled CI straddles zero → exonerated",
      li.ci_low < 0 < li.ci_high, (li.ci_low, li.ci_high))
check("CONTROL: verdict CLEAN", li.severity == SEV_CLEAN)
check("CONTROL: says the naive correlation was length tracking quality",
      "tracking quality" in li.detail)
check("CONTROL: finding is heuristic, cannot gate", li.confidence == CONF_HEURISTIC)

res = PROBES["self_preference"](str(FIX / "self-preference/judge_neutral.jsonl"),
                                gold_path=str(FIX / "self-preference/human_labels.jsonl"),
                                judge_family="gemini")
un = find(res, "bias.self.uninvolved-judge")
check("CONTROL: uninvolved judge → self-preference structurally impossible",
      un is not None and un.severity == SEV_CLEAN)
check("CONTROL: framed as the recommended setup, not a data gap",
      "recommended setup" in un.detail)

# =============================================================================
section("8. Refusing to over-claim")

res = PROBES["self_preference"](str(FIX / "self-preference/judge.jsonl"))
ng = find(res, "bias.self.no-gold")
check("no gold → refuses to call a raw score gap 'bias'", ng is not None)
check("no gold → says so in as many words",
      "not* evidence of bias" in ng.detail or "not evidence" in ng.detail.replace("*", ""))
check("no gold → names the confound as structural, not fixable with data",
      "more data will not resolve it" in ng.detail)
check("no gold → severity stays MINOR, never gates a build",
      ng.severity == SEV_MINOR)

res = PROBES["length_confound"](str(FIX / "probes/length_biased_judge.jsonl"))
check("no gold → length probe reports the naive r and calls it worthless",
      "worthless" in find(res, "bias.length.no-gold").detail)

res = PROBES["position"](str(FIX / "probes/verbosity.jsonl"))
check("wrong variants → probe explains what it needs instead of crashing",
      find(res, "bias.position.no-variants") is not None)

res = detect_drift(str(FIX / "drift/run_clean_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"))
# Outputs differ everywhere here, so there are no anchors to be had.
na = find(res, "drift.no-anchors")
check("no anchors → refuses to apportion the delta", na is not None)
check("no anchors → explains that splitting it would assume the answer",
      na is not None and "assuming the judge didn't move" in na.detail)
check("no anchors → prescribes the judge canary",
      na is not None and "Freeze ~30 items" in na.fix)

# =============================================================================
section("9. Loader — tolerant of real eval logs, strict about ids")

recs, rep = load_judge_records(str(FIX / "messy/nested.jsonl"))
check("reads nested outputs.score via flattening", len(recs) == 60 and recs[0].score is not None)
check("reads `key` as item_id", recs[0].item_id == "ex-000")
check("recovers fingerprint from metadata.judge_model",
      recs[0].fingerprint.model == "gpt-4o")
check("reports what it guessed", "score" in rep.describe())

recs, _ = load_judge_records(str(FIX / "messy/different_names.csv"))
check("reads CSV with foreign column names", len(recs) == 60)
check("maps q_id → item_id, rating → score",
      recs[0].item_id == "ex-000" and recs[0].score is not None)
check("maps evaluator_model → fingerprint",
      "claude" in recs[0].fingerprint.model)

recs, _ = load_judge_records(str(FIX / "messy/wrapped.json"))
check("reads a JSON array wrapped in {'results': [...]}", len(recs) == 60)
check("coerces boolean `passed` to a label",
      recs[0].label in ("pass", "fail"))

try:
    load_judge_records(str(FIX / "messy/no_id.jsonl"))
    check("no id → raises rather than pairing by row order", False, "no error raised")
except LoadError as e:
    check("no id → raises rather than pairing by row order", True)
    check("...and the error names the columns it did find", "score" in str(e), str(e))

recs, rep = load_judge_records(str(FIX / "drift/run_v1_nofingerprint.jsonl"))
check("missing fingerprint is warned about at load time",
      any("cannot be proven comparable" in w for w in rep.warnings), rep.warnings)

# Duplicate ids must be excluded, not silently de-duped.
pairs, warns = pair_runs(recs[:5] + recs[:5], recs[:5])
check("duplicate ids don't silently double-count", len(pairs) == 5)

# =============================================================================
section("10. emit_probe_set — the offline round trip")

import tempfile                                                 # noqa: E402
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "v.jsonl"
    dest, n = emit_probe_set(str(FIX / "probes/verbosity.jsonl"), "verbosity", str(out))
    check("verbosity variants emitted", n == 400, n)   # 200 in → 2 out each
    body = out.read_text()
    check("padded variants are tagged", '"variant": "padded"' in body)
    check("filler adds no information, only words", "It's worth noting at the outset" in body)
    check("scores are left null for your judge to fill", '"score": null' in body)

    out2 = Path(td) / "p.jsonl"
    _, n2 = emit_probe_set(str(FIX / "probes/position.jsonl"), "position", str(out2))
    check("position variants emitted in both orders", '"variant": "swapped"' in out2.read_text())

    out3 = Path(td) / "s.jsonl"
    emit_probe_set(str(FIX / "probes/sycophancy.jsonl"), "sycophancy", str(out3),
                   hint="new_model")
    check("sycophancy hint is content-free", "our new model" in out3.read_text())

    try:
        emit_probe_set(str(FIX / "probes/verbosity.jsonl"), "distribution", str(out))
        check("emit refuses probes that need no variants", False)
    except ValueError as e:
        check("emit refuses probes that need no variants", "need no variants" in str(e))

# =============================================================================
section("11. MCP tools — the surface an agent actually calls")

out = srv.detect_judge_drift(str(FIX / "drift/run_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"))
check("detect_judge_drift renders", "INVALIDATING" in out and "judge" in out.lower())
check("...and leads with the headline", "+6.0%" in out)

out = srv.calibrate_judge(str(FIX / "calibration/judge.jsonl"),
                          str(FIX / "calibration/human_labels.jsonl"))
check("calibrate_judge renders κ", "κ = 0.381" in out, out[:200])
check("...with the baseline beside it", "78.5%" in out)

out = srv.bias_probe(str(FIX / "probes/position.jsonl"), "position")
check("bias_probe renders", "contradicts itself" in out)

out = srv.bias_probe(str(FIX / "probes/verbosity.jsonl"), "nonsense")
check("unknown probe lists the real ones", "Available:" in out)

out = srv.audit_judge(str(FIX / "probes/distribution_collapsed.jsonl"),
                      scale_min=1, scale_max=10)
check("audit_judge runs every applicable probe", "distribution" in out)
check("...and says what's missing without gold labels", "no gold_path" in out)

out = srv.emit_probe_set(str(FIX / "probes/verbosity.jsonl"), "verbosity",
                         str(Path(tempfile.gettempdir()) / "e.jsonl"))
check("emit_probe_set tells you the next step", "bias_probe(" in out)

out = srv.explain_metric("kappa")
check("explain_metric(kappa) explains the base-rate trap", "78%" in out)
out = srv.explain_metric("anchors")
check("explain_metric(anchors) prescribes the canary", "judge canary" in out)
out = srv.explain_metric("nope")
check("explain_metric rejects unknown metrics", "Unknown metric" in out)

out = srv.calibrate_judge("/nonexistent.jsonl", "/also-missing.jsonl")
check("missing files fail gracefully, not with a traceback", out.startswith("⚠️"))

# =============================================================================
section("12. CLI — the CI gate")

def run_cli(args):
    return subprocess.run([sys.executable, "-m", "judge_audit_mcp.cli"] + args,
                          capture_output=True, text=True, cwd=str(ROOT))

r = run_cli(["drift", str(FIX / "drift/run_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"),
             "--fail-on", "serious"])
check("CLI fails the build on judge drift", r.returncode == 1, r.stderr[-200:])
check("...and says which finding did it", "judge" in r.stderr.lower())

r = run_cli(["drift", str(FIX / "drift/run_clean_v1.jsonl"),
             str(FIX / "drift/run_clean_v2.jsonl"), "--fail-on", "serious"])
check("CLI passes a clean run", r.returncode == 0, r.stderr[-200:])

r = run_cli(["probe", "length_confound", str(FIX / "probes/length_innocent_judge.jsonl"),
             "--gold", str(FIX / "probes/length_innocent_human.jsonl"),
             "--fail-on", "minor"])
check("CLI does not fail a build on a heuristic finding", r.returncode == 0, r.stderr[-200:])

r = run_cli(["drift", str(FIX / "drift/run_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"),
             "--format", "json"])
import json as _json                                            # noqa: E402
check("CLI --format json emits valid JSON", r.returncode in (0, 1))
doc = _json.loads(r.stdout)
check("...with counts and findings", "counts" in doc and doc["findings"])
check("...and the fingerprint that produced them", doc["fingerprint"]["model"])

r = run_cli(["drift", str(FIX / "drift/run_v1.jsonl"), str(FIX / "drift/run_v2.jsonl"),
             "--format", "sarif"])
doc = _json.loads(r.stdout)
check("CLI --format sarif emits SARIF 2.1.0", doc["version"] == "2.1.0")
check("...with rules attached", doc["runs"][0]["tool"]["driver"]["rules"])
check("...at error level for invalidating findings",
      any(x["level"] == "error" for x in doc["runs"][0]["results"]))

r = run_cli(["calibrate", str(FIX / "calibration/judge.jsonl"),
             str(FIX / "calibration/human_labels.jsonl"), "--fail-on", "serious"])
check("CLI calibrate gates on low κ", r.returncode == 1)

r = run_cli(["calibrate", str(FIX / "calibration/judge_good.jsonl"),
             str(FIX / "calibration/human_labels.jsonl"), "--fail-on", "serious"])
check("CLI calibrate passes a good judge", r.returncode == 0, r.stderr[-200:])

r = run_cli(["drift", "/nope.jsonl", "/nope2.jsonl"])
check("CLI exits 2 on a bad path (not 0, not a traceback)", r.returncode == 2)

# =============================================================================
section("13. Live adapter — off unless explicitly enabled")

from judge_audit_mcp import live                                 # noqa: E402
import os                                                        # noqa: E402

os.environ.pop("JUDGE_AUDIT_LIVE", None)
check("live judging is off by default", not live.is_enabled())
try:
    live.get_caller("anthropic", "claude-sonnet-5")
    check("live calls refuse without the env flag", False)
except live.LiveDisabled as e:
    check("live calls refuse without the env flag", True)
    check("...and the message points back to the offline path",
          "offline" in str(e))

check("score parser reads a bare integer", live.parse_score("7", 1, 10) == 7)
check("score parser reads JSON", live.parse_score('{"score": 8}', 1, 10) == 8)
check("score parser reads a chatty reply", live.parse_score("I'd say 9/10.", 1, 10) == 9)
check("score parser ignores out-of-range numbers",
      live.parse_score("Rating: 47", 1, 10) is None)
check("score parser returns None rather than guessing",
      live.parse_score("unsure", 1, 10) is None)

# =============================================================================
print("\n" + "=" * 72)
print(f"  {PASS} passed, {FAIL} failed")
print("=" * 72 + "\n")
sys.exit(1 if FAIL else 0)
