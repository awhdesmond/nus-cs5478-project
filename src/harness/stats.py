"""The statistics the proposal asks for (§4.2, Metrics): Wilson intervals and an exact McNemar test.

Both are short enough to write out, which also shows what they do.
"""

import math


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a success rate (z = 1.96).

    The plain "p ± z·√(p(1-p)/n)" interval breaks down exactly where results get interesting: at 0 % or 100 %
    it has zero width (20/20 successes doesn't prove the rate is exactly 100 %), and it can stray outside
    [0, 1]. Wilson's interval inverts the test instead: it's the range of true rates p for which the observed
    count wouldn't be surprising. For 20/20 it gives [0.84, 1.0].
    """
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes (both controllers ran the same seeds).

    Trials both solved, or both failed, say nothing about which controller is better; only the discordant
    pairs do: a_only = seeds only A solved, b_only = seeds only B solved. If the controllers were equally good,
    each discordant pair would be a fair coin flip, so the count a_only would be Binomial(n, ½) with
    n = a_only + b_only. The p-value is the chance of a split at least this lopsided.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
