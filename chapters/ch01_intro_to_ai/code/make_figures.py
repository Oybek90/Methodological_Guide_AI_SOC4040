# ci: skip
"""Generate the data-driven figures of Chapter 1 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paradigms import reinforcement  # noqa: E402
from subgroup_audit import SETTINGS, experiment  # noqa: E402
from tiny_lm import CORPUS, next_token_probs, train_bigram  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. Q-learning learning curve (mean over 20 seeds) ---------------------------------
curves = np.array([reinforcement(seed=s)[1] for s in range(20)], dtype=float)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(range(1, 41), curves.mean(axis=0)[:40], "k-", lw=1.2, label="mean steps per episode")
ax.axhline(7, color="gray", ls="--", lw=0.8, label="shortest path (7 steps)")
ax.set_xlabel("episode"); ax.set_ylabel("steps to reach the goal"); ax.legend(frameon=False, fontsize=8)
save(fig, "rl_curve.pdf")
print("mean steps ep1, ep10, ep40:", curves.mean(axis=0)[[0, 9, 39]].round(1))

# 2. Subgroup audit ----------------------------------------------------------------------
labels, a, b = [], [], []
for name, kw in SETTINGS.items():
    m = np.array([experiment(s, **kw) for s in range(20)]).mean(axis=0)
    labels.append(name.replace(", ", ",\n")); a.append(m[0]); b.append(m[1])
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.bar(x - w / 2, a, w, color="white", edgecolor="black", label="group A (majority)")
ax.bar(x + w / 2, b, w, color="white", edgecolor="black", hatch="///", label="group B (minority)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylim(0.5, 1.02)
ax.set_ylabel("test accuracy (20 seeds)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "subgroup_audit.pdf")

# 3. Bigram probabilities after "the" -------------------------------------------------
lm = train_bigram(CORPUS)
p = next_token_probs(lm, "the")
fig, ax = plt.subplots(figsize=(4.4, 2.6))
ax.barh(list(p)[::-1], list(p.values())[::-1], color="white", edgecolor="black")
ax.set_xlabel(r"$P(\mathrm{next\ word}\mid \mathrm{the})$")
save(fig, "bigram_probs.pdf")
print("done")
