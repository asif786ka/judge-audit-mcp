# judge-audit-mcp

[![PyPI](https://img.shields.io/pypi/v/judge-audit-mcp)](https://pypi.org/project/judge-audit-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/judge-audit-mcp)](https://pypi.org/project/judge-audit-mcp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/asif786ka/judge-audit-mcp/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-171%20passing-brightgreen)](https://github.com/asif786ka/judge-audit-mcp/blob/main/smoke_test.py)

**Your eval scores went up 6%. Nobody checked whether the judge changed.**

An MCP server that audits the LLM judge grading your evals: whether it agrees
with humans, whether it drifted between runs, and whether it's biased. Everyone
runs LLM-as-judge. Platforms ship judges. **None of them ship judge QA.**

```
$ judge-audit drift run_v1.jsonl run_v2.jsonl --fail-on serious

🛑 Judge changed between runs — 1 field(s) differ
    These two runs were scored by different instruments. Their means are not on
    the same scale, so the difference between them is not an improvement or a
    regression — it is not a quantity at all until the judge's own shift is
    measured and removed.

🛑 85% of the +6.0% score change is the judge, not the system
    Headline: 6.873 → 7.287 (+6.0%) over 150 common item(s).
    On 40 anchor item(s) — items whose output text is byte-identical in both
    runs, so the system provably did not change — the judge scored +0.350
    (+5.1%) higher in run B.
    That movement is the judge's, by construction. Removing it leaves +0.063
    for your system.
    The judge accounts for 85% of the headline.
    The improvement is the judge.
    delta (judge shift on anchors): +0.350  95% CI [+0.200, +0.500]  n=40

✗ failing: 2 finding(s) at or above serious
```

That's the whole product. Someone bumped the judge model between two eval runs.
Nobody wrote it down, because **the judge isn't the thing under test** — its
version doesn't belong in the changelog of the system being evaluated. The 6%
went in the launch doc. The system moved 0.9%.

This failure is silent (no error, the number is *plausible*), universal (everyone
bumps judge models — the alternative is pinning a deprecated one forever), and
unfalsifiable after the fact unless you happened to log the right things.

---

## The idea: a score is comparable only within a fingerprint

This is the third of three servers built on one idea, and the idea is that
**a claim isn't true, it's true within a scope** — so model the scope, and the
tool writes itself:

| server | a claim is valid within… | the method |
|---|---|---|
| [mobile-docs-mcp](https://github.com/asif786ka/mobile-docs-mcp) | a **version range** | `VersionRange.status_at(version)` |
| [store-preflight-mcp](https://github.com/asif786ka/store-preflight-mcp) | a **date window** | `PolicyWindow.status_on(date)` |
| **judge-audit-mcp** | a **judge fingerprint** | `JudgeFingerprint.comparable_to(other)` |

Same model, third axis. A judge score is not a measurement of the thing being
judged. It's a measurement of the thing being judged *as read by a particular
judge under a particular rubric* — and the moment any part of that changes, the
scale moves underneath you and the numbers stop being commensurable.

```python
JudgeFingerprint(
    model        = "claude-sonnet-5-20260401",  # "we upgraded" is a scale change
    prompt_hash  = "35c8032d",                  # PR: "clarify judge prompt"
    rubric_hash  = "ad8d493e",                  # someone tightened one criterion
    scale        = "1-10",
    temperature  = 0.0,                         # nonzero = not even repeatable
    provider     = "anthropic",
)
```

There's no "minor" field in there on purpose. Every one of them is something
people change *without thinking of it as changing the measuring instrument*.
That's precisely why it goes unnoticed.

**"Did my scores improve?" has no answer. It has an answer under a fixed
fingerprint.**

---

## Anchors: why the decomposition is an identity, not a model

You can't attribute a score change by comparing two means. `mean(B) − mean(A)`
confounds three things: the system changed, the eval set changed, and the judge
changed.

Joining on `item_id` kills the second. The third needs an **anchor set**: items
whose output is byte-identical across both runs. For those, the system *provably*
did not change, so any movement in their scores is the judge — wholly, and by
construction. That's not an assumption or a regression model. It's a difference
between two measurements of the same object.

```
judge_shift  = mean(score_B − score_A)  over anchors     [system held fixed]
system_shift = total_shift − judge_shift
```

Without anchors, this tool tells you the fingerprint changed and then **refuses
to apportion the delta**, because apportioning it would assume the thing in
question. That refusal is the honest answer, and the fix costs about a dollar:

> **Freeze ~30 items and their outputs. Re-judge them every run.**
> That's your judge canary. Almost nobody has one, which is why almost nobody
> catches judge drift.

---

## Accuracy is the trap

```
$ judge-audit calibrate judge.jsonl human_labels.jsonl

⚠️  Judge–human agreement: Cohen's κ = 0.381 (fair)
    Raw agreement 83.0%, but 72.5% of that is expected by chance given the base
    rates. A judge that always said 'pass' would score 78.5% on this set.
    Cohen's κ: 0.381  95% CI [+0.219, +0.535]  n=200

⚠️  Judge beats a constant-answer baseline by only +4.5%
    83.0% accuracy sounds respectable until you notice that always answering
    'pass' scores 78.5% on this set. The judge is contributing +4.5% over a rule
    that reads none of the input.

ℹ️  Judge is 0.349 below the human ceiling
    Humans agree with each other at κ=0.730; the judge reaches κ=0.381. There is
    real headroom here — this gap belongs to the judge, not to the task.
```

83% accurate reads fine in a slide. κ = 0.38 says it's barely doing better than
a rule that reads none of the input. On any realistic eval set — where most items
pass — **accuracy flatters a broken judge**, and it does so exactly when the
benchmark is most representative.

The third finding is the one most tools skip. If your annotators only agree with
*each other* at κ=0.55, a judge at κ=0.50 is at the noise floor of the task and
"improve the judge" is the wrong project — the rubric is underdefined and no
judge, human or otherwise, can do better. Reporting κ without the ceiling sends
teams on month-long goose chases to fix a judge that's already as good as a
person. Pass `--human-ceiling`, or log `rater_agreement` in your label file, and
this tool will tell you which problem you actually have.

---

## Six bias probes

Each asks: **does the verdict move when something that shouldn't matter changes?**
The manipulation is orthogonal to quality by construction, so any response to it
is bias by definition — there's no "well, maybe it was right" argument to have.

| probe | manipulation | needs |
|---|---|---|
| `position` | swap A/B order | swapped variants |
| `verbosity` | pad with content-free filler | padded variants |
| `self_preference` | own-family vs other, **vs a human panel** | `output_family` + gold |
| `length_confound` | length **at equal human quality** | gold + lengths |
| `sycophancy` | a leading hint in the prompt | hinted variants |
| `distribution` | *(none — reads your existing log)* | **nothing** |

`bias_probe(path, probe='distribution')` costs zero extra judge calls and runs on
a log you already have. It's also the one that invalidates the most results:

```
🛑 Judge's effective resolution is 2.5 level(s), not 10
    The judge used 3 distinct values: 7×62, 8×189, 9×49. Entropy 1.32 bits.
    Everyone reading a score 'out of 10' believes they are looking at ten rungs
    of measurement. They are looking at 2.5.

⚠️  7 rubric level(s) never used: 1, 2, 3, 4, 5, 6, 10
    Whatever your rubric says those levels mean has never influenced a score.

ℹ️  Your +0.430 delta is 129 item(s) hopping one notch
    This judge emits whole numbers only. A +0.430 change is not 300 items each
    improving slightly — it is ~129 items crossing a single threshold while the
    other 171 did not move at all.
```

### The probe that proves the others aren't rubber stamps

Anyone can write a bias detector that always finds bias. The interesting question
is whether it can be made to shut up.

`length_confound` exists to survive the obvious objection to `verbosity`: *"of
course the judge scores long answers higher — long answers ARE better. Your bias
is the judge being right."* On observational data, a correlation cannot refute
that. So control for quality with the human score:

```
judge_score ~ b0 + b1·human_score + b2·length
```

`b2` is what the judge pays for length **among answers humans rated equally**. If
length were merely a proxy for quality, b2 ≈ 0 — once you know how good an answer
is, its length tells you nothing more, and there's nothing left to pay for.

Two fixtures ship with this repo. Both have a near-identical naive correlation —
the number people quote as proof of bias. The controlled slope separates them:

| | naive `r(length, judge)` | controlled `b2` | verdict |
|---|---|---|---|
| `length_biased_judge` | **+0.815** | +1.454, CI [+0.70, +2.21] | 🛑 pays for length |
| `length_innocent_judge` | **+0.775** | +0.389, CI [−0.30, +1.11] | ✅ exonerated |

Same story from the naive number. Opposite truths. That's the whole argument for
the probe — and every scenario in this repo ships with a control fixture like
this, generated by the same mechanism with the bias removed. The tests assert
both that the tool fires *and* that it stays quiet.

---

## Findings carry their own epistemics

Inherited from store-preflight's confidence tiers, and load-bearing here.

The bias literature reports effects that are **real but modest** — position flips
in the 10-30% range, self-preference well under a point on a 10-point scale.
Effects that size are trivially manufactured by sampling noise at n=20 plus a
hopeful analyst. A tool that reports "κ = 0.61, substantial!" from 14 items is
worse than no tool: it launders noise into a number with a Greek letter on it and
someone puts it in a slide.

So every finding carries `n`, a bootstrap CI, and a tier that **downgrades
itself**:

- `certain` — mechanical. *"The judge model string differs between these runs"*
  is a string comparison, not an inference, and needs no sample at all. The most
  damaging finding this tool makes is also the only one that needs no statistics.
- `likely` — the interval excludes the null **and** n clears the power floor.
- `heuristic` — underpowered or the CI straddles null. **Never gates a build**,
  whatever its severity.

`temper_confidence()` is the direct analogue of `PolicyWindow.severity_on()`: the
claim isn't hand-edited as evidence accumulates, the model does it. A tool that
red-builds on an underpowered estimate gets switched off within a week — and then
it audits nothing at all.

---

## Install

```bash
uvx judge-audit-mcp          # MCP server (stdio)
pip install judge-audit-mcp  # library + CLI
```

One dependency (`mcp`). The statistics are pure stdlib — Cohen's κ, weighted κ,
bootstrap intervals, exact binomial tests, OLS. scipy+sklearn would be ~80MB of
wheels to compute a 2×2 contingency table, and an audit tool that's annoying to
install doesn't get installed.

That trade is only defensible if the reimplementations are demonstrably right, so
the smoke test cross-checks **every estimator** against scipy/scikit-learn when
they're present and skips those assertions when they aren't:

```bash
pip install judge-audit-mcp[verify]
python smoke_test.py     # 171 passing (160 without scipy — cross-checks skip)
```

The dependency is optional for *using* this server and mandatory for *doubting*
it, which is the correct way round.

### Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "judge-audit": {
      "command": "uvx",
      "args": ["judge-audit-mcp"]
    }
  }
}
```

---

## Tools

| tool | question |
|---|---|
| `calibrate_judge` | Does the judge agree with humans beyond chance? (κ, baseline, ceiling) |
| `detect_judge_drift` | Did my system improve, or did the judge change underneath me? |
| `bias_probe` | Is the judge biased? (six probes) |
| `emit_probe_set` | Generate the variants a probe needs me to judge |
| `audit_judge` | Run everything this data supports; say what's missing for the rest |
| `explain_metric` | What is κ / an anchor / a fingerprint, and why is it the right question? |

### CI gate

```yaml
- run: judge-audit drift baseline.jsonl current.jsonl --fail-on serious
- run: judge-audit calibrate judge.jsonl gold.jsonl --fail-on serious
```

`--format sarif` renders findings as inline PR annotations on GitHub. Judge drift
becomes a red build instead of a wrong number in a launch doc.

---

## Your data probably already works

The loader is aggressively tolerant, because everyone's eval log has a different
shape and a tool that demands a schema audits nobody's judge. JSONL, JSON arrays,
wrapped objects, CSV/TSV; nested fields flattened; `key`/`q_id`/`example_id` all
read as ids; `outputs.score`/`rating`/`grade` all read as scores; booleans coerced
to labels.

Guessing is always **reported**, never silent — reading `latency_ms` as your score
column and confidently reporting κ would be worse than failing.

One firm rule: **`item_id` is required and never invented.** Row order is not an
identifier, and pairing by it would fabricate the very join the conclusions rest
on.

---

## The offline round trip

The probes need variants. This server doesn't call your judge — it writes the file
of things to judge, you run your own judge over it, and you feed the results back:

```bash
judge-audit emit position my_eval.jsonl        # → my_eval.position.jsonl
# ... score those with your judge ...
judge-audit probe position my_eval.position.jsonl
```

That round trip is why the core needs no API key, costs nothing, and produces the
same numbers every time. An audit tool whose own results move between runs is in
no position to lecture anyone about reproducibility.

A live adapter exists behind `JUDGE_AUDIT_LIVE=1` + an API key for demos. It's a
convenience. The offline path is the product.

---

## The fixtures are generated, not written

```bash
python fixtures/generate.py
```

Every fixture comes out of a simulated process with a **stated mechanism**, not
out of someone tuning a JSON file until the test went green. The generator says
what it simulates and why the resulting number is the number — e.g. the drift
fixture's judge upgrades ~30% of items by one point, and the +6.0% headline falls
out of that. Change the mechanism and the headline moves; the tool's attribution
tracks it either way, which is the property under test.

Seeded, so the files reproduce byte for byte. If a fixture ever needs to change,
the diff shows the *mechanism* changing — the only honest way to edit a test
fixture.

---

## What this doesn't do

- **It can't tell you your judge is good.** It can tell you it disagrees with your
  humans, drifted, or responds to things it shouldn't. A judge that passes every
  probe here may still be wrong in ways nobody has thought to probe for.
- **It inherits your labels' biases.** `length_confound` measures the judge's
  excess length bias *over your annotators'*. If your humans are length-biased
  too, it under-reports.
- **`self_preference` and `length_confound` need human labels.** Without them
  they report the confounded number and say, in as many words, that it isn't
  evidence. That's a refusal, not a limitation to be worked around — judge scores
  alone cannot separate "favours its own" from "its own was better", and more
  data will never fix it.
- **Nonzero-temperature judges aren't repeatable at all**, and this tool can only
  flag that, not quantify it, unless you supply repeat runs.

---

## References

- Cohen (1960), *A Coefficient of Agreement for Nominal Scales* — κ
- Landis & Koch (1977) — the interpretation bands (a convention, not a law)
- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — position & verbosity bias
- Wang et al. (2023), *Large Language Models are not Fair Evaluators* — order effects
- Panickssery et al. (2024), *LLM Evaluators Recognize and Favor Their Own Generations* — self-preference
- Dubois et al. (2024), *Length-Controlled AlpacaEval* — length control
- Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models*

## License

MIT
