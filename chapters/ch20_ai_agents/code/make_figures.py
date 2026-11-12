# ci: skip
"""Figures for Chapter 20 (uses reliability_sim.py and agent_loop.py)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import comb
from reliability_sim import pipeline_success, majority_indep, majority_corr, speedup

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))

ks = np.arange(1, 11)
for q, ls, lab in ((0, "ko-", "no reviewer"), (0.3, "ks--", "reviewer, $q=0.3$"), (0.7, "k^:", "reviewer, $q=0.7$")):
    ax[0].plot(ks, [pipeline_success(.9, k, q) for k in ks], ls, mfc="w", ms=4, label=lab)
ax[0].set_xlabel("stages $k$ (each right w.p. 0.9)"); ax[0].set_ylabel("P(whole pipeline right)")
ax[0].set_ylim(0, 1.02); ax[0].set_title("A. chain of agents"); ax[0].legend(frameon=False, fontsize=7)

ns = np.arange(1, 26, 2)
for rho, ls, lab in ((0, "ko-", r"independent"), (0.3, "ks--", r"$\rho=0.3$"), (0.6, "k^:", r"$\rho=0.6$")):
    ax[1].plot(ns, [majority_corr(n, .7, rho) for n in ns], ls, mfc="w", ms=4, label=lab)
ax[1].set_xlabel("voters $n$ (each right w.p. 0.7)"); ax[1].set_ylabel("P(majority right)")
ax[1].set_ylim(0.6, 1.02); ax[1].set_title("B. majority vote"); ax[1].legend(frameon=False, fontsize=7)

ms = np.arange(1, 17)
ax[2].plot(ms, [speedup(m) for m in ms], "ko-", ms=4, label="speed-up")
ax[2].axhline(1, color="gray", lw=0.8)
ax[2].set_xlabel("parallel sub-agents $m$"); ax[2].set_ylabel("speed-up over one agent")
ax[2].set_title("C. parallel workers")
a2 = ax[2].twinx(); a2.spines["right"].set_visible(True)
a2.plot(ms, ms + 1, "k--", lw=0.8); a2.set_ylabel("token cost (x one agent)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_reliability.pdf"), bbox_inches="tight")
