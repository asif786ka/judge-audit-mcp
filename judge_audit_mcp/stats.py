"""Statistics for judge auditing. Pure standard library, no numpy, no scipy.

Why dependency-free, when scipy exists and is correct?

Because the other two servers in this trilogy declare exactly one dependency
(`mcp`) and `uvx judge-audit-mcp` should stay a cold-start-in-a-second
proposition rather than a 90MB scientific-Python download. An audit tool that is
annoying to install does not get installed, and a judge that never gets audited
is the entire problem this server exists to fix. numpy+scipy is ~80MB of wheels
to compute a 2x2 contingency table.

The honest cost of that choice: these implementations must be *demonstrably*
right, or the trade is a bad one. So `smoke_test.py` cross-checks every estimator
here against scikit-learn and scipy when they happen to be installed, and skips
those checks when they aren't. The dependency is optional for *using* this
server and mandatory for *doubting* it, which is the correct way round.

Design note on inference: there is no t-distribution or normal approximation
anywhere in this file. Every interval is a bootstrap percentile interval and
every p-value is an exact binomial/sign test. That's a deliberate trade —
slightly wider intervals in exchange for no distributional assumptions about
judge scores, which are bounded, discrete, lumpy, and wildly non-normal (a
1-10 judge that only ever emits 7 or 8 violates every assumption a t-test makes).
Bootstrapping a median-ish quantity from clustered scores is exactly the case
where the textbook approximation embarrasses you.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence

# Fixed seed everywhere. Two runs of an audit on identical data must produce
# identical numbers, or the audit tool has the same reproducibility problem as
# the thing it is auditing — which would be funny once and useless forever.
DEFAULT_SEED = 20260715
DEFAULT_BOOTSTRAP = 2000


# --- basics ------------------------------------------------------------------

def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation (n-1)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")   # zero variance: correlation is undefined, not zero.
    return num / (dx * dy)


def cohens_d_paired(deltas: Sequence[float]) -> float:
    """Standardised effect size for a paired difference.

    Reported alongside raw deltas because "the judge scores padded answers 0.4
    higher" means nothing until you know whether 0.4 is a rounding error or half
    the spread of the entire scale.
    """
    sd = stdev(deltas)
    if not sd or math.isnan(sd) or sd == 0:
        return float("nan")
    return mean(deltas) / sd


# --- agreement ---------------------------------------------------------------

def observed_agreement(a: Sequence, b: Sequence) -> float:
    if not a:
        return float("nan")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def chance_agreement(a: Sequence, b: Sequence) -> float:
    """Expected agreement if both raters guessed independently at their own base rates."""
    n = len(a)
    if not n:
        return float("nan")
    cats = set(a) | set(b)
    pa = {c: sum(1 for x in a if x == c) / n for c in cats}
    pb = {c: sum(1 for x in b if x == c) / n for c in cats}
    return sum(pa[c] * pb[c] for c in cats)


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's κ for two raters over categorical labels.

        κ = (po - pe) / (1 - pe)

    The whole point of κ over raw accuracy: it subtracts off the agreement you'd
    get by luck given each rater's base rate. On a benchmark where 78% of items
    pass, a judge that says "pass" unconditionally scores 78% accuracy and κ = 0.
    Accuracy flatters a broken judge exactly when the eval set is realistic,
    which is to say always.

    Degenerate case worth naming: when both raters use exactly one category and
    it's the same one, pe = 1 and κ is 0/0. Convention here is 1.0 (they agreed
    on everything), but a κ of 1.0 from a constant rater is not good news, so
    callers should look at the distribution findings too. This is a known
    pathology of κ, not a bug in this function.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    po = observed_agreement(a, b)
    pe = chance_agreement(a, b)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def weighted_kappa(a: Sequence, b: Sequence, weights: str = "linear") -> float:
    """Cohen's weighted κ for *ordinal* scores.

        κ_w = 1 - Σ(w_ij · O_ij) / Σ(w_ij · E_ij)

    Use this, not plain κ, whenever scores are ordered. Plain κ treats a judge
    that says 9 where the human said 8 as exactly as wrong as one that says 1 —
    which on a 1-10 rubric is absurd and will make a perfectly decent judge look
    broken.

    Categories are the sorted unique values across both raters, weighted by
    *index distance*, matching sklearn's `cohen_kappa_score(weights=...)`. The
    caveat that buys: if a level is never used by either rater it doesn't exist
    in the grid, so index distance can diverge from value distance (scores of
    1,2,10 are treated as equally spaced). Judge scales are usually dense enough
    that this is fine, and matching sklearn means the cross-check in the smoke
    test is meaningful rather than approximate.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    levels = sorted(set(a) | set(b))
    k = len(levels)
    if k < 2:
        return 1.0 if observed_agreement(a, b) == 1.0 else 0.0
    idx = {v: i for i, v in enumerate(levels)}

    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[idx[x]][idx[y]] += 1.0

    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]

    def w(i: int, j: int) -> float:
        d = abs(i - j) / (k - 1)
        return d * d if weights == "quadratic" else d

    num = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * E[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - num / den


def majority_baseline(labels: Sequence) -> tuple[float, object]:
    """Accuracy of the dumbest possible judge: always guess the most common label.

    This number belongs next to every accuracy claim about a judge. It is the
    bar, and "81% accurate" clears it by 3 points on a set that's 78% pass.
    """
    if not labels:
        return float("nan"), None
    counts: dict = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    top, c = max(counts.items(), key=lambda kv: kv[1])
    return c / len(labels), top


def prevalence(labels: Sequence) -> dict:
    n = len(labels)
    if not n:
        return {}
    counts: dict = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    return {k: v / n for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


# --- intervals ---------------------------------------------------------------

def bootstrap_ci(
    stat_fn: Callable[[Sequence[int]], float],
    n: int,
    iters: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI over *paired indices*.

    `stat_fn` takes a list of indices (a resample) and returns the statistic.
    Resampling indices rather than values is what keeps pairs intact — for
    position bias the two orderings of one comparison must be resampled together
    or the dependence is destroyed and the interval comes out far too narrow,
    manufacturing significance out of nothing.
    """
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            v = stat_fn(idx)
        except (ZeroDivisionError, ValueError):
            continue
        if v is not None and not math.isnan(v):
            stats.append(v)
    if len(stats) < 20:
        return (float("nan"), float("nan"))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return (lo, hi)


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the normal approximation because flip rates live near 0 and 1
    where the textbook interval famously runs off the end of the scale and
    reports a negative lower bound for a probability.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def mean_delta_ci(
    deltas: Sequence[float],
    iters: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    ds = list(deltas)
    return bootstrap_ci(lambda idx: mean([ds[i] for i in idx]), len(ds), iters, alpha, seed)


# --- exact tests -------------------------------------------------------------

def binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test.

    Method of small p-values: sum the probability of every outcome no more likely
    than the observed one. Exact rather than approximate because judge audits
    routinely run on n=20 paired swaps, where the normal approximation is a work
    of fiction.
    """
    if n == 0:
        return float("nan")
    if not 0 < p < 1:
        return float("nan")
    obs = binom_pmf(k, n, p)
    tot = 0.0
    for i in range(n + 1):
        pi = binom_pmf(i, n, p)
        if pi <= obs * (1 + 1e-9):
            tot += pi
    return min(1.0, tot)


def sign_test(deltas: Sequence[float]) -> tuple[float, int, int]:
    """Exact sign test on paired differences. Returns (p, n_pos, n_nonzero).

    Zeros are dropped, which is the standard convention. Worth knowing that this
    makes the test conservative and deliberately blind to effect *size* — it only
    asks "does the judge move in one direction more often than a coin would?".
    That's the right question for a bias probe: a verbosity effect that is tiny
    but always in the same direction is a real bias, and one that is large but
    signless is noise.
    """
    nz = [d for d in deltas if d != 0]
    if not nz:
        return (1.0, 0, 0)
    pos = sum(1 for d in nz if d > 0)
    return (binom_test_two_sided(pos, len(nz), 0.5), pos, len(nz))


# --- regression --------------------------------------------------------------

def _solve(A: list[list[float]], b: list[float]) -> Optional[list[float]]:
    """Gaussian elimination with partial pivoting. Returns None if singular."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / pv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def ols(X: list[list[float]], y: list[float]) -> Optional[list[float]]:
    """Ordinary least squares via normal equations. X excludes the intercept
    column; one is prepended here. Returns [b0, b1, ...] or None if singular.

    Normal equations rather than QR: the design matrices here have two or three
    columns of well-scaled data, so the conditioning objection to X'X doesn't
    bite, and the whole thing is 30 lines instead of a linear algebra library.
    """
    n = len(y)
    if n == 0 or len(X) != n:
        return None
    k = len(X[0]) + 1
    if n <= k:
        return None   # No residual degrees of freedom; the fit would be vacuous.
    D = [[1.0] + list(row) for row in X]
    XtX = [[sum(D[r][i] * D[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(D[r][i] * y[r] for r in range(n)) for i in range(k)]
    return _solve(XtX, Xty)


def ols_slope_ci(
    X: list[list[float]],
    y: list[float],
    coef: int = 1,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Bootstrap CI for one OLS coefficient.

    Bootstrapping cases rather than deriving analytic standard errors, because
    the analytic version needs homoskedastic normal residuals and judge-score
    residuals are neither. Also keeps this file free of a t-distribution CDF.
    """
    def stat(idx: Sequence[int]) -> float:
        b = ols([X[i] for i in idx], [y[i] for i in idx])
        return float("nan") if b is None else b[coef]
    return bootstrap_ci(stat, len(y), iters, alpha, seed)


def r_squared(X: list[list[float]], y: list[float], beta: list[float]) -> float:
    n = len(y)
    if n == 0 or not beta:
        return float("nan")
    my = mean(y)
    sst = sum((v - my) ** 2 for v in y)
    if sst == 0:
        return float("nan")
    sse = 0.0
    for r in range(n):
        pred = beta[0] + sum(beta[i + 1] * X[r][i] for i in range(len(X[r])))
        sse += (y[r] - pred) ** 2
    return 1.0 - sse / sst


# --- distribution pathology --------------------------------------------------

def entropy_bits(values: Sequence) -> float:
    n = len(values)
    if not n:
        return float("nan")
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def effective_levels(values: Sequence) -> float:
    """Perplexity of the score distribution: how many levels the judge *behaves*
    as if it has, as opposed to how many the rubric says it has.

    This is the number that kills "we improved from 7.2 to 7.6 on a 1-10 scale".
    If the judge only ever emits 7, 8, and 9 — and emits 8 most of the time —
    its effective resolution is ~2, and a 0.4 move is a handful of items hopping
    one notch on a scale with three usable rungs. Reported as a continuous
    quantity because a judge that uses level 3 twice in 500 items has not really
    got a level 3.
    """
    h = entropy_bits(values)
    return float("nan") if math.isnan(h) else 2 ** h


def ceiling_share(values: Sequence[float], top: Optional[float] = None) -> float:
    """Fraction of scores sitting at the maximum. A judge with 60% of its mass on
    the top score cannot measure improvement — there's nowhere up to go, so real
    gains vanish into a ceiling and the eval reports nothing."""
    if not values:
        return float("nan")
    hi = max(values) if top is None else top
    return sum(1 for v in values if v >= hi) / len(values)


def unused_levels(values: Sequence[float], scale_min: float, scale_max: float,
                  step: float = 1.0) -> list[float]:
    """Rubric levels the judge never once used."""
    seen = set(values)
    out, v = [], scale_min
    while v <= scale_max + 1e-9:
        if v not in seen:
            out.append(v)
        v += step
    return out
