# ci: skip
"""Generate the figures of Chapter 3 into ../figures/ (vector PDF, grayscale-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from multireg import X_RAW, Y, add_bias, gradient_descent, standardise  # noqa: E402
from polyreg import (fit_ols, fit_ridge, make_data, poly_features, split_fit,  # noqa: E402
                     true_curve)

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# 1. Gradient descent on the lecture data: raw vs standardised features ---------------------
A = add_bias(X_RAW)
_, h_raw = gradient_descent(A, Y, lr=0.001, iters=20000)
Az = add_bias(standardise(X_RAW)[0])
_, h_z = gradient_descent(Az, Y, lr=0.1, iters=20000)
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(h_raw, "k--", lw=1.0, label=r"raw features, $\alpha=10^{-3}$")
ax.plot(h_z, "k-", lw=1.0, label=r"standardised, $\alpha=0.1$")
ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel(r"cost $J$ (log scale)")
ax.legend(frameon=False, fontsize=8)
save(fig, "gd_multi.pdf")
print("J after 20000: raw", f"{h_raw[-1]:.4f}", " standardised", f"{h_z[-1]:.2e}")

# 2. Three polynomial fits to one training set -------------------------------------------------------
rng = np.random.default_rng(0)
x, y = make_data(30, 3.0, rng)
xs = np.linspace(-3, 3, 300)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, d in zip(axs, (1, 3, 9)):
    mu = poly_features(x, d)[:, 1:].mean(axis=0); sd = poly_features(x, d)[:, 1:].std(axis=0)

    def feats(v, d=d, mu=mu, sd=sd):
        F = poly_features(v, d); F[:, 1:] = (F[:, 1:] - mu) / sd; return F

    w = fit_ols(feats(x), y)
    ax.scatter(x, y, s=12, color="0.35", zorder=3)
    ax.plot(xs, true_curve(xs), "0.6", ls="--", lw=1.0, label="true curve")
    ax.plot(xs, feats(xs) @ w, "k-", lw=1.1, label="fitted")
    ax.set_title(f"degree {d}", fontsize=9); ax.set_xlabel("x"); ax.set_ylim(-25, 30)
axs[0].set_ylabel("y"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "poly_fits.pdf")

# 3. Train/test MSE against degree (median over 200 training sets) --------------------------
degs = list(range(1, 13))
res = {d: np.array([split_fit(d, seed=s) for s in range(200)]) for d in degs}
tr = [np.median(res[d][:, 0]) for d in degs]
te = [np.median(res[d][:, 1]) for d in degs]
q1 = [np.percentile(res[d][:, 1], 25) for d in degs]
q3 = [np.percentile(res[d][:, 1], 75) for d in degs]
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(degs, tr, "k--o", ms=3, lw=1.0, label="train MSE (median)")
ax.plot(degs, te, "k-o", ms=3, lw=1.0, label="test MSE (median)")
ax.fill_between(degs, q1, q3, color="0.8", label="test MSE, quartiles")
ax.axhline(9, color="0.4", ls=":", lw=0.9); ax.text(11.9, 9.6, "noise variance 9", ha="right", fontsize=7.5)
ax.set_yscale("log"); ax.set_xlabel("polynomial degree"); ax.set_ylabel("MSE (log scale)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "degree_curve.pdf")
for d in (1, 2, 3, 4, 5, 9, 12):
    print("degree", d, "train", round(tr[d - 1], 2), "test median", round(te[d - 1], 2),
          "IQR", round(q1[d - 1], 2), round(q3[d - 1], 2))
print("best degree by median test MSE:", degs[int(np.argmin(te))])

# 4. Ridge on degree 9 -----------------------------------------------------------------------------
lams = np.logspace(-3, 2, 11)
rr = {l: np.array([split_fit(9, seed=s, lam=l) for s in range(200)]) for l in lams}
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(lams, [np.median(rr[l][:, 0]) for l in lams], "k--o", ms=3, lw=1.0, label="train MSE (median)")
ax.plot(lams, [np.median(rr[l][:, 1]) for l in lams], "k-o", ms=3, lw=1.0, label="test MSE (median)")
ax.axhline(9, color="0.4", ls=":", lw=0.9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"regularisation strength $\lambda$"); ax.set_ylabel("MSE (log scale)")
ax.legend(frameon=False, fontsize=7.5)
save(fig, "ridge_lambda.pdf")
best = min(lams, key=lambda l: np.median(rr[l][:, 1]))
print("best lambda (median test MSE):", round(best, 3), round(np.median(rr[best][:, 1]), 2))
