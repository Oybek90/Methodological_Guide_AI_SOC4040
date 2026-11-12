# ci: skip
"""Figures for Chapter 19 (run mini_gpt.py and scaling_laws.py first)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from attention_np import sinusoidal_pe, softmax
from scaling_laws import FITS, loss, optimum

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
save = lambda f, n: (f.savefig(os.path.join(OUT, n), bbox_inches="tight"), plt.close(f))

# 1. sinusoidal positional encoding and the 1/sqrt(d_k) scale ------------------------------------
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
pe = sinusoidal_pe(64, 32)
im = ax[0].imshow(pe.T, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax[0].set_xlabel("position $p$"); ax[0].set_ylabel("dimension $j$"); ax[0].set_title("sinusoidal $PE_{p,j}$")
sim = pe @ pe.T
ax[1].imshow(sim, cmap="viridis"); ax[1].set_title("$PE_p\\cdot PE_q$ (depends on $p-q$)")
ax[1].set_xlabel("$q$"); ax[1].set_ylabel("$p$")
dks = [4, 16, 64, 256, 1024, 4096]
m0, m1 = [], []
for dk in dks:
    q = rng.normal(size=(2000, dk)); k = rng.normal(size=(10, dk)); s = q @ k.T
    m0.append(softmax(s).max(1).mean()); m1.append(softmax(s / np.sqrt(dk)).max(1).mean())
ax[2].semilogx(dks, m0, "ko-", label="no scaling"); ax[2].semilogx(dks, m1, "ks--", mfc="w", label="$/\\sqrt{d_k}$")
ax[2].set_xlabel("$d_k$"); ax[2].set_ylabel("mean largest weight"); ax[2].set_ylim(0, 1.05)
ax[2].set_title("10 keys, random $q,k$"); ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig_pe_scale.pdf")

# 2. mini-GPT: training curves and attention ---------------------------------------------------------
R = json.load(open(os.path.join(HERE, "mini_gpt_results.json")))
log = np.array(R["log"])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(log[:, 0], log[:, 1], "k-", label="train"); ax.plot(log[:, 0], log[:, 2], "k--", label="validation")
for k, ls in (("1-gram", ":"), ("2-gram", "-."), ("3-gram", ":"), ("4-gram", "-.")):
    ax.axhline(R["baselines"][k], color="0.6", lw=0.8, ls=ls)
    ax.text(1510, R["baselines"][k], k, va="center", fontsize=7.5, color="0.35")
ax.set_xlim(0, 1700); ax.set_ylim(1.1, 3.2)
ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy (nats per character)"); ax.legend(frameon=False)
save(fig, "fig_mini_gpt_loss.pdf")

Z = np.load(os.path.join(HERE, "mini_gpt_attn.npz")); att, txt = Z["att"], str(Z["text"])
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2))
for h, a in enumerate(axs):
    a.imshow(att[h], cmap="Greys", vmin=0, vmax=1)
    a.set_title(f"head {h + 1}", fontsize=8)
    a.set_xticks(range(len(txt))); a.set_xticklabels(list(txt), fontsize=5)
    a.set_yticks(range(len(txt))); a.set_yticklabels(list(txt), fontsize=5)
    a.tick_params(length=0)
axs[0].set_ylabel("query (row)"); fig.supxlabel("key (column)", fontsize=8)
fig.tight_layout(); save(fig, "fig_gpt_attn.pdf")

# 3. scaling laws -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
Ns = np.logspace(7, 12.5, 100)
for C, ls in ((1e20, ":"), (1e22, "--"), (1e24, "-")):
    ax[0].loglog(Ns, C / (6 * Ns), "0.7", ls=ls, lw=0.9)
for name, fit, st in (("published fit", FITS["Hoffmann et al. (published)"], "k--"),
                      ("replication", FITS["Besiroglu et al. (replication)"], "k-")):
    Cs = np.logspace(19, 26, 50); nd = np.array([optimum(c, fit) for c in Cs])
    ax[0].loglog(nd[:, 0], nd[:, 1], st, lw=1.2, label=f"optimum, {name}")
offs = {"GPT-3": (-4, -10), "Gopher": (5, -3), "Chinchilla": (-48, 6), "LLaMA-65B": (4, 6)}
for k, (N, D) in {"GPT-3": (175e9, 300e9), "Gopher": (280e9, 300e9), "Chinchilla": (70e9, 1.4e12), "LLaMA-65B": (65e9, 1.4e12)}.items():
    ax[0].plot(N, D, "ko", ms=3.5); ax[0].annotate(k, (N, D), offs[k], textcoords="offset points", fontsize=7)
ax[0].set_xlim(1e8, 3e12); ax[0].set_ylim(1e9, 1e14)
ax[0].set_xlabel("parameters $N$"); ax[0].set_ylabel("training tokens $D$"); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fit = FITS["Besiroglu et al. (replication)"]
for D, ls in ((2e10, ":"), (2e11, "--"), (2e12, "-")):
    ax[1].loglog(Ns, loss(Ns, D, fit) - fit[0], "k" + ls if ls != "-" else "k-", lw=1.1, label=f"$D=${D:.0e}")
ax[1].set_xlabel("parameters $N$"); ax[1].set_ylabel("reducible loss $L-E$"); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("replication fit: $A/N^\\alpha+B/D^\\beta$", fontsize=8)
fig.tight_layout(); save(fig, "fig_scaling.pdf")
