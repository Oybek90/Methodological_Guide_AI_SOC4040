# ci: run
"""Why (and when) several agents help: three small, exactly checkable models.

A. a pipeline of k agents, optionally with a reviewer that catches an error and triggers one retry
B. majority vote of n agents with independent or correlated errors
C. wall-clock speed-up of m parallel sub-agents with coordination overhead
"""
import numpy as np
from math import comb


def pipeline_success(p, k, q=0.0):
    """Each stage is right w.p. p. With a reviewer (detects an error w.p. q) the stage is
    redone once (right again w.p. p). Whole pipeline right only if all stages right."""
    stage = p + (1 - p) * q * p
    return stage ** k


def pipeline_mc(p, k, q, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.ones(n, bool)
    for _ in range(k):
        first = rng.random(n) < p
        caught = (~first) & (rng.random(n) < q)
        retry = rng.random(n) < p
        ok &= first | (caught & retry)
    return ok.mean()


def majority_indep(n, p):
    """P(more than half of n independent voters are right); odd n."""
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n // 2 + 1, n + 1))


def majority_corr(n, p, rho):
    """With prob rho all voters share one outcome (a common cause, e.g. same model, same
    misleading source); otherwise independent."""
    return rho * p + (1 - rho) * majority_indep(n, p)


def majority_mc(n, p, rho, trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.random(trials) < rho
    shared = rng.random(trials) < p
    votes = rng.random((trials, n)) < p
    maj = votes.sum(1) > n // 2
    return np.where(common, shared, maj).mean()


def speedup(m, overhead=0.15, trials=20000, seed=0):
    """m sub-tasks with lognormal durations (mean ~1). Sequential = sum. Parallel =
    overhead*m (planning/merging grows with m) + slowest sub-task."""
    rng = np.random.default_rng(seed)
    d = rng.lognormal(-0.125, 0.5, (trials, m))
    seq = d.sum(1)
    par = overhead * m + d.max(1)
    return seq.mean() / par.mean()


if __name__ == "__main__":
    print("A. pipeline of k stages, p = 0.9 per stage")
    for k in (1, 3, 5, 10):
        print(f"  k={k:2d}: no reviewer {pipeline_success(.9, k):.3f} (mc {pipeline_mc(.9, k, 0):.3f})"
              f" | reviewer q=0.7 {pipeline_success(.9, k, .7):.3f} (mc {pipeline_mc(.9, k, .7):.3f})"
              f" | reviewer q=0.3 {pipeline_success(.9, k, .3):.3f}")
    assert abs(pipeline_success(.9, 3) - 0.729) < 1e-12
    assert abs(pipeline_success(.9, 3, .7) - pipeline_mc(.9, 3, .7)) < 0.005

    print("B. majority vote, single-agent accuracy p = 0.7")
    print("   n   independent   rho=0.3   rho=0.6   (mc rho=0.3)")
    for n in (1, 3, 5, 9, 25):
        print(f"  {n:2d}   {majority_indep(n, .7):.3f}        {majority_corr(n, .7, .3):.3f}     "
              f"{majority_corr(n, .7, .6):.3f}     {majority_mc(n, .7, .3):.3f}")
    assert abs(majority_indep(3, .7) - 0.784) < 1e-9
    assert abs(majority_indep(5, .7) - 0.83692) < 1e-9
    assert abs(majority_corr(9, .7, .3) - majority_mc(9, .7, .3)) < 0.005
    print("   limit n->inf, independent: 1.0; rho=0.3: ", round(0.3 * .7 + 0.7 * 1, 3),
          " rho=0.6:", round(0.6 * .7 + 0.4, 3))

    print("C. speed-up of m parallel sub-agents (overhead 0.15 per sub-task)")
    for m in (1, 2, 4, 8, 16):
        print(f"  m={m:2d}: speed-up {speedup(m):.2f}   token cost x{m + 1} (m workers + 1 coordinator, equal size)")
    assert speedup(1) < 1.0 < speedup(4)
    print("reliability_sim self-test passed")
