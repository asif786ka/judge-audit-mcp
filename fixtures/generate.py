"""Generate the fixture sets.

Committed as a generator rather than by hand so the fixtures are *honest*: the
numbers in the README come out of a simulated process with a stated mechanism,
not out of reverse-engineering a nice-looking κ. Every scenario below says what
it simulates and why the resulting number is the number.

Seeded, so `python fixtures/generate.py` reproduces the committed files byte for
byte. If a fixture ever needs to change, the diff shows the mechanism changing —
which is the only honest way to edit a test fixture.

    python fixtures/generate.py
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent

FILLER_PRE = ("It's worth noting at the outset that this is a question that benefits "
              "from careful consideration, and there are a number of angles from which "
              "one might approach it. With that context in mind, here is the response:\n\n")
FILLER_SUF = ("\n\nTo summarise the above: the key points have been laid out, and taken "
              "together they address the question as posed. It is of course worth "
              "bearing in mind that individual circumstances vary, and there may be "
              "additional nuances depending on the specifics of the situation. Overall, "
              "the considerations described above should provide a useful basis for "
              "thinking about this.")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}: {len(rows)} rows")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =============================================================================
# 1. drift/ — "the 6% that wasn't"
# =============================================================================
# Mechanism, stated exactly:
#   - 150 eval items. 40 of them are ANCHORS: frozen outputs, byte-identical in
#     both runs. The system did not touch them.
#   - Between run 1 and run 2 the team bumped the judge model. Nothing else about
#     the judge changed. Nobody wrote this down anywhere except the eval log.
#   - The new judge is systematically more generous: it upgrades ~30% of items by
#     one point. That is the ONLY thing that changed for the anchor items.
#     (30% is a parameter of the simulation, chosen so the headline lands near the
#     +6% of the canonical story. Change it and the headline moves; the tool's
#     attribution tracks it either way, which is the property under test.)
#   - The system itself genuinely improved on the 110 non-anchor items, but only
#     barely — a real but tiny gain.
#
# The point of the fixture: the headline is ~+6%, and almost all of it is the
# judge. A team looking at the headline ships "we improved 6%". The anchors say
# the judge got +5.x% more generous on outputs that did not change at all.

def gen_drift() -> None:
    rng = random.Random(4242)
    d = ROOT / "drift"

    fp_v1 = {"judge_model": "claude-3-5-sonnet-20241022", "provider": "anthropic",
             "temperature": 0.0, "scale": "1-10",
             "judge_prompt": "Grade the response 1-10 for correctness and clarity.",
             "rubric": "10=flawless, 7=good with minor issues, 4=partially wrong, 1=useless"}
    # Only the model string differs. Same prompt, same rubric, same scale, same temp.
    fp_v2 = dict(fp_v1, judge_model="claude-sonnet-5-20260401")

    run_a, run_b = [], []
    anchors = []

    for i in range(150):
        iid = f"item-{i:03d}"
        is_anchor = i < 40
        text = f"Response to question {i}. " + "Detail. " * rng.randint(3, 20)

        base = clamp(round(rng.gauss(7.0, 1.3)), 1, 10)

        # Judge v2 is more generous: bumps ~30% of items by one point, ceiling at 10.
        judged_higher = rng.random() < 0.30
        b_score = clamp(base + (1 if judged_higher else 0), 1, 10)

        if is_anchor:
            # Frozen: same output text both runs. Any score change is the judge.
            anchors.append(iid)
            a_text = b_text = text
            a_score = base
        else:
            # System changed: output changed, and genuinely improved a little — 12% of
            # items gain a real point on top of whatever the judge does.
            a_text = text
            b_text = text + " Additionally, note the following clarification."
            a_score = base
            if rng.random() < 0.12:
                b_score = clamp(b_score + 1, 1, 10)

        run_a.append({"item_id": iid, "score": a_score, "output_text": a_text, **fp_v1})
        run_b.append({"item_id": iid, "score": b_score, "output_text": b_text, **fp_v2})

    write(d / "run_v1.jsonl", run_a)
    write(d / "run_v2.jsonl", run_b)
    (d / "anchors.txt").write_text(",".join(anchors) + "\n", encoding="utf-8")

    # --- the same runs, with no fingerprint logged at all ---------------------
    # The most common real-world state: you cannot even ask the question.
    strip = lambda r: {k: v for k, v in r.items()
                       if k not in ("judge_model", "provider", "temperature", "scale",
                                    "judge_prompt", "rubric")}
    write(d / "run_v1_nofingerprint.jsonl", [strip(r) for r in run_a])
    write(d / "run_v2_nofingerprint.jsonl", [strip(r) for r in run_b])

    # --- honest control: same judge both runs, system genuinely improved ------
    # The tool must NOT cry wolf here. A drift detector that fires on real
    # improvements is worse than none, because it teaches you to ignore it.
    ca, cb = [], []
    rng2 = random.Random(99)
    for i in range(150):
        iid = f"item-{i:03d}"
        base = clamp(round(rng2.gauss(7.0, 1.3)), 1, 10)
        text = f"Response to question {i}. " + "Detail. " * rng2.randint(3, 20)
        is_anchor = i < 40
        b = base if is_anchor else clamp(base + (1 if rng2.random() < 0.4 else 0), 1, 10)
        ca.append({"item_id": iid, "score": base, "output_text": text, **fp_v1})
        cb.append({"item_id": iid, "score": b,
                   "output_text": text if is_anchor else text + " Clarified.", **fp_v1})
    write(d / "run_clean_v1.jsonl", ca)
    write(d / "run_clean_v2.jsonl", cb)


# =============================================================================
# 2. calibration/ — "the judge that flips a coin"
# =============================================================================
# Mechanism:
#   - 200 items, binary pass/fail. Base rate is deliberately realistic: 78% pass.
#   - The judge is right on ~81% of items — a number that reads as fine.
#   - But its errors are structured the way a lazy judge's are: it almost never
#     calls a fail (it says "pass" on most of the genuine fails), and its
#     agreement is therefore mostly the base rate doing the work.
#   - Humans agree with each other at κ≈0.73, so the ceiling is high: this is NOT
#     an underdefined task. The judge really is the problem.
#
# The point: 81% accurate, and κ lands in the fair-to-moderate band, barely above
# a constant "pass" that scores 78% while reading nothing.

def gen_calibration() -> None:
    rng = random.Random(1337)
    d = ROOT / "calibration"
    judge, gold = [], []

    fp = {"judge_model": "gpt-4o-mini", "provider": "openai", "temperature": 0.0,
          "scale": "binary",
          "judge_prompt": "Did the response correctly answer the question? pass/fail.",
          "rubric": "pass if factually correct and responsive; fail otherwise"}

    for i in range(200):
        iid = f"q-{i:03d}"
        truth = "pass" if rng.random() < 0.78 else "fail"
        # The judge's failure mode: heavily biased toward "pass".
        if truth == "pass":
            jv = "pass" if rng.random() < 0.93 else "fail"   # rarely fails a good one
        else:
            jv = "fail" if rng.random() < 0.38 else "pass"   # misses most real failures
        judge.append({"item_id": iid, "label": jv, **fp})
        gold.append({"item_id": iid, "label": truth, "n_raters": 3,
                     "rater_agreement": 0.73})

    write(d / "judge.jsonl", judge)
    write(d / "human_labels.jsonl", gold)

    # --- a well-calibrated judge on the same items, as the contrast -----------
    good = []
    rng3 = random.Random(555)
    for g in gold:
        truth = g["label"]
        jv = truth if rng3.random() < 0.93 else ("fail" if truth == "pass" else "pass")
        good.append({"item_id": g["item_id"], "label": jv, **fp})
    write(d / "judge_good.jsonl", good)

    # --- the ceiling case: judge is mediocre, but so are the humans -----------
    # Same judge quality as `judge.jsonl`, but inter-human κ is only 0.42 — the
    # task is underdefined and no judge can do better. The tool must say so
    # rather than sending someone off to "fix the judge".
    ceil_gold = [dict(g, rater_agreement=0.42) for g in gold]
    write(d / "human_labels_noisy.jsonl", ceil_gold)


# =============================================================================
# 3. self-preference/ — "the judge that likes itself"
# =============================================================================
# Mechanism:
#   - 180 items, half produced by a claude-family model, half by a gpt-family one.
#   - GROUND TRUTH: the claude outputs are genuinely slightly better (+0.3 on a
#     10-point scale, per the human panel). This is deliberate and load-bearing —
#     it's what makes the naive test look right for the wrong reason.
#   - The judge is claude-family. On top of the real quality gap it adds a further
#     +0.8 to its own family and −0.1 to the other, relative to humans.
#
# The point: a naive mean comparison shows claude ahead by ~1.2 and someone
# writes "claude wins". Some of that is real. Most of it is the judge marking its
# own homework. Only the residual against humans separates the two.

def gen_self_preference() -> None:
    rng = random.Random(31415)
    d = ROOT / "self-preference"
    judge, gold = [], []

    fp = {"judge_model": "claude-sonnet-5-20260401", "provider": "anthropic",
          "temperature": 0.0, "scale": "1-10",
          "judge_prompt": "Grade the response 1-10.",
          "rubric": "10=flawless, 1=useless"}

    for i in range(180):
        iid = f"pair-{i:03d}"
        fam = "claude" if i % 2 == 0 else "gpt"
        # True quality: claude genuinely a bit better.
        true_q = rng.gauss(6.6 if fam == "claude" else 6.3, 1.2)
        human = clamp(round(true_q), 1, 10)
        # Judge = truth + self-preference + noise.
        bias = 0.8 if fam == "claude" else -0.1
        js = clamp(round(true_q + bias + rng.gauss(0, 0.5)), 1, 10)
        text = f"Answer {i} from {fam}. " + "Content. " * rng.randint(4, 18)
        judge.append({"item_id": iid, "score": js, "output_family": fam,
                      "output_text": text, **fp})
        gold.append({"item_id": iid, "human_score": human, "n_raters": 3,
                     "rater_agreement": 0.68})

    write(d / "judge.jsonl", judge)
    write(d / "human_labels.jsonl", gold)

    # --- control: a gemini judge on the same outputs, no dog in the fight -----
    # Same true quality, no family bias. The tool must find nothing here — this
    # is what "use a judge from an uninvolved family" looks like when it works.
    neutral = []
    rng4 = random.Random(27182)
    for j, g in zip(judge, gold):
        true_q = g["human_score"] + rng4.gauss(0, 0.4)
        neutral.append({"item_id": j["item_id"], "score": clamp(round(true_q), 1, 10),
                        "output_family": j["output_family"], "output_text": j["output_text"],
                        **dict(fp, judge_model="gemini-2.0-pro", provider="google")})
    write(d / "judge_neutral.jsonl", neutral)


# =============================================================================
# 4. probes/ — position, verbosity, sycophancy, distribution
# =============================================================================

def gen_probes() -> None:
    d = ROOT / "probes"

    # --- position ------------------------------------------------------------
    # Mechanism: 120 pairwise comparisons. The judge has a genuine preference on
    # 70% of them (it picks the same content in both orders). On the other 30% it
    # contradicts itself, and those contradictions are NOT random — they favour
    # whatever is presented first 80% of the time. That's the distinction the
    # probe is built to draw: noise cancels, skew accumulates.
    rng = random.Random(8080)
    fp = {"judge_model": "gpt-4o", "provider": "openai", "temperature": 0.0,
          "scale": "pairwise", "judge_prompt": "Which response is better, A or B?",
          "rubric": "pick the more helpful response"}
    rows = []
    for i in range(120):
        iid = f"cmp-{i:03d}"
        if rng.random() < 0.70:
            # Consistent: same content wins both times → opposite slots.
            c1 = rng.choice(["A", "B"])
            c2 = "B" if c1 == "A" else "A"
        else:
            # Contradiction: same slot both times. Skewed toward first position.
            c1 = c2 = "A" if rng.random() < 0.80 else "B"
        rows.append({"item_id": iid, "variant": "", "choice": c1, **fp})
        rows.append({"item_id": iid, "variant": "swapped", "choice": c2, **fp})
    write(d / "position.jsonl", rows)

    # Control: a consistent judge. 96% consistent, contradictions unskewed.
    rng_c = random.Random(606)
    rows = []
    for i in range(120):
        iid = f"cmp-{i:03d}"
        if rng_c.random() < 0.96:
            c1 = rng_c.choice(["A", "B"])
            c2 = "B" if c1 == "A" else "A"
        else:
            c1 = c2 = rng_c.choice(["A", "B"])
        rows.append({"item_id": iid, "variant": "", "choice": c1, **fp})
        rows.append({"item_id": iid, "variant": "swapped", "choice": c2, **fp})
    write(d / "position_clean.jsonl", rows)

    # --- verbosity -----------------------------------------------------------
    # Mechanism: 100 outputs judged as-written and padded with content-free
    # filler. The judge pays ~0.6 points for the padding, ~75% of the time.
    rng = random.Random(2024)
    fp = {"judge_model": "claude-sonnet-5-20260401", "provider": "anthropic",
          "temperature": 0.0, "scale": "1-10", "judge_prompt": "Grade 1-10.",
          "rubric": "10=flawless, 1=useless"}
    rows = []
    for i in range(100):
        iid = f"ans-{i:03d}"
        base = clamp(round(rng.gauss(6.5, 1.4)), 1, 10)
        text = f"Answer {i}. " + "Substance. " * rng.randint(4, 15)
        padded_text = FILLER_PRE + text + FILLER_SUF
        lift = 1 if rng.random() < 0.75 else 0
        rows.append({"item_id": iid, "variant": "", "score": base,
                     "output_text": text, "output_len": len(text), **fp})
        rows.append({"item_id": iid, "variant": "padded", "score": clamp(base + lift, 1, 10),
                     "output_text": padded_text, "output_len": len(padded_text), **fp})
    write(d / "verbosity.jsonl", rows)

    # --- sycophancy ----------------------------------------------------------
    # Mechanism: 90 outputs judged neutrally and with "this is from our new
    # model". The judge moves up on ~55% of them.
    rng = random.Random(1234)
    rows = []
    for i in range(90):
        iid = f"ans-{i:03d}"
        base = clamp(round(rng.gauss(6.5, 1.3)), 1, 10)
        text = f"Answer {i}. " + "Content. " * rng.randint(4, 12)
        bump = 1 if rng.random() < 0.55 else 0
        rows.append({"item_id": iid, "variant": "", "score": base,
                     "output_text": text, **fp})
        rows.append({"item_id": iid, "variant": "hinted", "score": clamp(base + bump, 1, 10),
                     "output_text": text, **fp})
    write(d / "sycophancy.jsonl", rows)

    # --- distribution --------------------------------------------------------
    # Mechanism: a 1-10 rubric where the judge only ever emits 7, 8, 9 — and 8
    # most of the time. Effective resolution ≈ 1.4 levels. Levels 1-6 and 10 are
    # decoration. This is not a caricature; it is what most 1-10 judges do.
    rng = random.Random(777)
    rows = []
    for i in range(300):
        s = rng.choices([7, 8, 9], weights=[0.22, 0.63, 0.15])[0]
        rows.append({"item_id": f"item-{i:03d}", "score": s,
                     "output_text": "x" * rng.randint(200, 900), **fp})
    write(d / "distribution_collapsed.jsonl", rows)

    # A judge that actually uses its scale, as the contrast.
    rng = random.Random(888)
    rows = [{"item_id": f"item-{i:03d}", "score": clamp(round(rng.gauss(5.5, 2.2)), 1, 10),
             "output_text": "x" * rng.randint(200, 900), **fp} for i in range(300)]
    write(d / "distribution_healthy.jsonl", rows)

    # A judge stuck on one value — usually a parsing bug, not a judgement.
    rows = [{"item_id": f"item-{i:03d}", "score": 8, "output_text": "x" * 300, **fp}
            for i in range(120)]
    write(d / "distribution_constant.jsonl", rows)

    # --- length confound -----------------------------------------------------
    # Two fixtures, and the contrast between them is the whole argument.
    #
    # (a) BIASED: length and quality are correlated (better answers really are
    #     longer), AND the judge pays an extra premium for length on top. The
    #     naive correlation is high; the controlled slope is also positive.
    rng = random.Random(4711)
    j, g = [], []
    for i in range(150):
        iid = f"item-{i:03d}"
        q = rng.gauss(5.5, 1.8)
        # Length genuinely tracks quality, plus noise.
        L = int(clamp(200 + q * 90 + rng.gauss(0, 120), 60, 1600))
        human = clamp(round(q), 1, 10)
        # Judge = quality + a real length premium (~1.1 pts per 1k chars).
        js = clamp(round(q + 1.1 * (L / 1000.0) + rng.gauss(0, 0.4)), 1, 10)
        j.append({"item_id": iid, "score": js, "output_text": "x" * L,
                  "output_len": L, **fp})
        g.append({"item_id": iid, "human_score": human, "rater_agreement": 0.7})
    write(d / "length_biased_judge.jsonl", j)
    write(d / "length_biased_human.jsonl", g)

    # (b) INNOCENT: length and quality are correlated exactly as before, but the
    #     judge pays NOTHING for length once quality is known. The naive
    #     correlation looks just as damning. The controlled slope is ~0.
    #     If the probe fires here, it's broken.
    rng = random.Random(1123)
    j, g = [], []
    for i in range(150):
        iid = f"item-{i:03d}"
        q = rng.gauss(5.5, 1.8)
        L = int(clamp(200 + q * 90 + rng.gauss(0, 120), 60, 1600))
        human = clamp(round(q), 1, 10)
        js = clamp(round(q + rng.gauss(0, 0.4)), 1, 10)   # no length term at all
        j.append({"item_id": iid, "score": js, "output_text": "x" * L,
                  "output_len": L, **fp})
        g.append({"item_id": iid, "human_score": human, "rater_agreement": 0.7})
    write(d / "length_innocent_judge.jsonl", j)
    write(d / "length_innocent_human.jsonl", g)


# =============================================================================
# 5. messy/ — schema tolerance
# =============================================================================
# Real eval logs from three different platforms, none agreeing on field names,
# one nested, one CSV. If the loader can't read these it audits nobody's judge.

def gen_messy() -> None:
    d = ROOT / "messy"
    rng = random.Random(31)

    # LangSmith-ish: nested outputs, `key` as id.
    rows = [{"key": f"ex-{i:03d}",
             "outputs": {"score": rng.randint(1, 10), "comment": "..."},
             "metadata": {"judge_model": "gpt-4o", "temperature": 0}}
            for i in range(60)]
    write(d / "nested.jsonl", rows)

    # CSV with entirely different column names.
    lines = ["q_id,rating,evaluator_model,response"]
    for i in range(60):
        lines.append(f"ex-{i:03d},{rng.randint(1, 10)},claude-sonnet-5,\"answer {i}\"")
    (d / "different_names.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {(d / 'different_names.csv').relative_to(ROOT)}: 60 rows")

    # JSON array wrapped in an object, booleans as labels.
    rows = {"results": [{"id": f"ex-{i:03d}", "passed": rng.random() < 0.7}
                        for i in range(60)]}
    (d / "wrapped.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"  {(d / 'wrapped.json').relative_to(ROOT)}: 60 rows")

    # A file with no id at all — must fail loudly rather than pair by row order.
    rows = [{"score": rng.randint(1, 10)} for i in range(20)]
    write(d / "no_id.jsonl", rows)


if __name__ == "__main__":
    print("Generating fixtures...")
    gen_drift()
    gen_calibration()
    gen_self_preference()
    gen_probes()
    gen_messy()
    print("Done.")
