# ci: skip
"""Figures for Chapter 17 (grayscale-safe vector PDFs). Run seq_models.py first (it writes seq_a.npz, seq_b.npz)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rnn_np import rnn_forward

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# 1. gradient flow: norm of d h_T / d h_t against the lag, vanilla RNN (H = 32) and LSTM forget-gate products ----
H, T = 32, 60
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
for rho, ls in ((0.5, ":"), (0.9, "--"), (1.5, "-")):
    rng = np.random.default_rng(1)
    Wh = rng.normal(size=(H, H)); Wh *= rho / max(abs(np.linalg.eigvals(Wh)))
    p = {"Wx": rng.normal(0, 0.5, (H, 1)), "Wh": Wh, "b": np.zeros(H), "Wy": np.zeros(H), "by": 0.0}
    hs, _ = rnn_forward(rng.normal(size=(T, 1)), p)
    M, norms = np.eye(H), []
    for k in range(T, 0, -1):
        norms.append(np.linalg.norm(M, 2))
        M = M @ (np.diag(1 - hs[k] ** 2) @ Wh)
    axs[0].semilogy(norms, "k" + ls, label=r"$\rho(W_h)=%.1f$" % rho)
axs[0].set_xlabel("lag $T-t$"); axs[0].set_ylabel(r"$\|\partial h_T/\partial h_t\|_2$"); axs[0].legend(frameon=False)
axs[0].set_title("vanilla RNN")
lag = np.arange(T)
for f, ls in ((0.5, ":"), (0.9, "--"), (0.99, "-")):
    axs[1].semilogy(lag, f ** lag, "k" + ls, label=f"$f={f}$")
axs[1].set_xlabel("lag $T-t$"); axs[1].set_ylabel(r"$\prod_k f_k$"); axs[1].legend(frameon=False)
axs[1].set_title("LSTM cell state, constant forget gate")
save(fig, "fig_gradient.pdf")

# 2. one-step-ahead prediction ----------------------------------------------------------------------------------
A = np.load(os.path.join(HERE, "seq_a.npz"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})
axs[0].plot(A["yte"], "k-", lw=1.6, label="actual")
for k, ls in (("RNN", ":"), ("LSTM", "--"), ("GRU", "-.")):
    axs[0].plot(A["pred_" + k], "k" + ls, lw=1.0, label=k)
axs[0].set_xlabel("test sample"); axs[0].set_ylabel("scaled value"); axs[0].legend(frameon=False, ncol=2)
for k, ls in (("RNN", ":"), ("LSTM", "--"), ("GRU", "-.")):
    axs[1].semilogy(A["loss_" + k], "k" + ls, label=k)
axs[1].set_xlabel("epoch"); axs[1].set_ylabel("training loss"); axs[1].legend(frameon=False)
save(fig, "fig_series.pdf")

# 3. memory task ---------------------------------------------------------------------------------------------------
B = np.load(os.path.join(HERE, "seq_b.npz"))
fig, ax = plt.subplots(figsize=(3.8, 2.8))
for k, ls, mk in (("RNN", ":", "o"), ("LSTM", "--", "s"), ("GRU", "-.", "^")):
    ax.errorbar(B["Ts"], B[k].mean(1), B[k].std(1), fmt="k" + ls + mk, capsize=2, ms=4, label=k)
ax.axhline(0.5, color="0.6", lw=0.8); ax.set_xlabel("sequence length $T$"); ax.set_ylabel("test accuracy")
ax.set_ylim(0.4, 1.05); ax.legend(frameon=False)
save(fig, "fig_memory.pdf")
print("figures written to", os.path.abspath(OUT))
