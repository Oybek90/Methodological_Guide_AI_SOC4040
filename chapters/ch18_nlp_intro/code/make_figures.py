# ci: skip
"""Figures for Chapter 18. Run tweets_experiment.py first (writes tweets_results.json)."""
import os
import re
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# 1. accuracy: leakage and representation ---------------------------------------------------
R = json.load(open(os.path.join(HERE, "tweets_results.json")))
keys = ["multi-hot, emoticons kept", "multi-hot, emoticons removed", "BoW (counts), emoticons removed",
        "TF-IDF, emoticons removed", "TF-IDF, keep stop words", "TF-IDF uni+bigrams", "GloVe-50 mean vector + LR"]
labels = ["multi-hot\n(emoticons kept)", "multi-hot", "BoW\n(no stop w.)", "TF-IDF\n(no stop w.)", "TF-IDF\n(stop w. kept)",
          "TF-IDF\nuni+bigrams", "GloVe-50\nmean"]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
m = [R[k][0] for k in keys]; s = [R[k][1] for k in keys]
hatch = ["//"] + [""] * 6
ax.bar(range(len(keys)), m, yerr=s, color=["0.75"] + ["0.45"] * 6, edgecolor="k", capsize=2)
for i, (v, h) in enumerate(zip(m, hatch)):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0.5, 1.05); ax.set_ylabel("test accuracy")
ax.axhline(0.5, color="k", lw=0.6, ls=":")
save(fig, "fig_tweets_acc.pdf")

# 2. Zipf's law on the tweet corpus ------------------------------------------------------------
import nltk
from nltk.corpus import twitter_samples
nltk.download("twitter_samples", quiet=True)
tw = twitter_samples.strings("positive_tweets.json") + twitter_samples.strings("negative_tweets.json")
cnt = Counter(w for t in tw for w in re.findall(r"[a-z']+", t.lower()))
freq = np.array(sorted(cnt.values(), reverse=True))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].loglog(np.arange(1, len(freq) + 1), freq, "k-", lw=1)
axs[0].set_xlabel("rank of the word"); axs[0].set_ylabel("frequency")
axs[0].set_title("rank-frequency (log-log)")
cum = np.cumsum(freq) / freq.sum()
axs[1].plot(np.arange(1, len(freq) + 1), cum, "k-")
axs[1].set_xscale("log"); axs[1].set_xlabel("vocabulary size (most frequent words first)")
axs[1].set_ylabel("share of all tokens"); axs[1].set_title("coverage")
for k in (100, 1000):
    axs[1].plot([k], [cum[k - 1]], "ko"); axs[1].annotate(f"{k}: {cum[k-1]:.0%}", (k, cum[k - 1]), (k * 1.3, cum[k - 1] - 0.08), fontsize=8)
print("vocab", len(freq), "hapax", int((freq == 1).sum()), "top100 cover", round(float(cum[99]), 3), "top1000", round(float(cum[999]), 3))
save(fig, "fig_zipf.pdf")

# 3. GloVe geometry: project onto a "gender" axis and a "capital" axis -------------------------
import gensim.downloader as api
kv = api.load("glove-wiki-gigaword-50")
pairs = [("man", "woman"), ("king", "queen"), ("uncle", "aunt"), ("brother", "sister")]
caps = [("france", "paris"), ("italy", "rome"), ("japan", "tokyo"), ("germany", "berlin")]
g = np.mean([kv[b] - kv[a] for a, b in pairs], axis=0)
c = np.mean([kv[b] - kv[a] for a, b in caps], axis=0)
g /= np.linalg.norm(g)
c -= (c @ g) * g; c /= np.linalg.norm(c)               # orthonormal pair (g, c)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
allw = [w for p in pairs + caps for w in p]
mu = np.mean([kv[w] for w in allw], axis=0)
for group, marker in ((pairs, "o"), (caps, "s")):
    for a, b in group:
        pa, pb = [((kv[w] - mu) @ g, (kv[w] - mu) @ c) for w in (a, b)]
        ax.annotate("", pb, pa, arrowprops=dict(arrowstyle="->", color="0.4"))
        for p, w in ((pa, a), (pb, b)):
            ax.plot(*p, "k" + marker, ms=4); ax.text(p[0] + 0.12, p[1] + 0.12, w, fontsize=8)
ax.set_xlabel("gender direction (mean of the four offsets)")
ax.set_ylabel("capital direction")
save(fig, "fig_glove_axes.pdf")
