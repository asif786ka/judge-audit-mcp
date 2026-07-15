"""Bias probes.

Each probe answers one question of the form "does the judge's verdict move when
something that should not matter changes?". That framing is what makes a probe a
*test* rather than a vibe: the manipulation is orthogonal to quality by
construction, so any response to it is bias by definition, and there's no
argument to have about whether the judge was "right".

    position       swap A/B order            → content is identical, so any flip is order
    verbosity      pad with filler           → padding adds no information
    self_preference own-family vs other      → measured against a human panel
    length_confound length vs quality        → controls for human score
    sycophancy     leading hint in prompt    → hint carries no evidence
    distribution   no manipulation           → can the scale even resolve your delta?

`distribution` is the odd one out — it needs no variants, only the judge's raw
scores. It's here because it's the cheapest probe (zero extra judge calls, runs
on any eval log you already have) and it invalidates more results than any of the
others: a 1-10 judge whose effective resolution is 2.1 levels cannot detect the
4% improvement you are reporting from it, and it will never say so.

Design rule for every probe: the manipulation must be *quality-preserving*, and
where it isn't obviously so, the probe says what it is assuming. Padding an
answer with "It's worth noting that..." doesn't make it better — but it does make
it longer, and `length_confound` exists precisely because "longer" and "better"
are correlated in real data, so a naive verbosity probe can't tell a biased judge
from a judge that's right about a real correlation.
"""
from __future__ import annotations

from .distribution import probe_distribution
from .length_confound import probe_length_confound
from .position import probe_position
from .self_preference import probe_self_preference
from .sycophancy import probe_sycophancy
from .verbosity import probe_verbosity

PROBES = {
    "position": probe_position,
    "verbosity": probe_verbosity,
    "self_preference": probe_self_preference,
    "length_confound": probe_length_confound,
    "sycophancy": probe_sycophancy,
    "distribution": probe_distribution,
}

# Probes that need nothing but a plain judge run — no variants, no human labels.
# Worth knowing which these are: they can be run on logs you already have, today,
# with no extra judge calls and no annotation budget.
FREE_PROBES = ("distribution",)

__all__ = ["PROBES", "FREE_PROBES", "probe_position", "probe_verbosity",
           "probe_self_preference", "probe_length_confound", "probe_sycophancy",
           "probe_distribution"]
