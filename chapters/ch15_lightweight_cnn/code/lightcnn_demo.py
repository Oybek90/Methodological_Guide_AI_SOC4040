# ci: run
"""Numbers quoted in Chapter 15 (parameters, multiply-adds, equivalences, batch normalization)."""
import numpy as np
from lightcnn import (params_grouped, macs_grouped, params_separable, macs_separable, grouped_conv2d,
                      depthwise_conv2d, pointwise_conv2d, BatchNorm2d, batchnorm_forward)
from cnn_numpy import conv2d

print("== 1. Slides 7-11: C = K = 8 channels, 3x3 filter, no bias")
for g in (1, 2, 8):
    print(f"groups={g}: {params_grouped(8, 8, 3, g, bias=False)} weights")
print("separable (depthwise g=8 + pointwise):", 8 * 9, "+", 8 * 8, "=", params_separable(8, 8, 3, bias=False))

print("\n== 2. 32 -> 64 channels, 3x3, with bias (Chapter 14 numbers)")
std, grp, sep = params_grouped(32, 64, 3), params_grouped(32, 64, 3, 32), params_separable(32, 64, 3)
print("standard", std, "grouped(32)", grp, "separable", sep, f"ratio {std / sep:.1f}")
w_std, w_sep = 32 * 64 * 9, 32 * 9 + 32 * 64
print(f"weights only: {w_std} vs {w_sep}: ratio {w_sep / w_std:.4f}; 1/K+1/F^2 = {1 / 64 + 1 / 9:.4f}")

print("\n== 3. Multiply-adds on a 16x16 output map")
ms, mp = macs_grouped(32, 64, 3, 16, 16), macs_separable(32, 64, 3, 16, 16)
print(f"standard {ms:,}  separable {mp:,}  ratio {ms / mp:.2f}")
for f in (3, 5, 7):
    print(f"F={f}: K=64 ratio standard/separable = {1 / (1 / 64 + 1 / f ** 2):.2f}")

print("\n== 4. Equivalences")
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 8, 9, 9))
wd, wp = rng.normal(size=(8, 1, 3, 3)), rng.normal(size=(8, 8, 1, 1))
y = pointwise_conv2d(depthwise_conv2d(x, wd, pad=1), wp)
print("separable output shape", y.shape)
w_full = rng.normal(size=(8, 8, 3, 3))
print("groups=1 equals standard:", np.allclose(grouped_conv2d(x, w_full, groups=1), conv2d(x, w_full)))
# a separable convolution is a standard convolution whose filters have rank-one channel structure
w_eq = wp[:, :, 0, 0][:, :, None, None] * wd[:, 0][None]       # (K, C, 3, 3)
print("separable == standard with filters W[k,c]=pw[k,c]*dw[c]:",
      np.allclose(conv2d(x, w_eq, pad=1), y))
print("1x1 conv with C=8, K=4:", params_grouped(8, 4, 1), "parameters")

print("\n== 5. Batch normalization")
xb = rng.normal(loc=5.0, scale=3.0, size=(16, 4, 6, 6))
yb, _, mu, var = batchnorm_forward(xb, np.ones(4), np.zeros(4))
print("input channel means", np.round(xb.mean(axis=(0, 2, 3)), 2), "stds", np.round(xb.std(axis=(0, 2, 3)), 2))
print("output channel means", np.round(yb.mean(axis=(0, 2, 3)), 6), "stds", np.round(yb.std(axis=(0, 2, 3)), 4))
bn = BatchNorm2d(4)
for _ in range(60):
    bn.forward(rng.normal(loc=5.0, scale=3.0, size=(16, 4, 6, 6)))
print("running mean", np.round(bn.run_mean, 2), "running var", np.round(bn.run_var, 1))
bn.training = False
xe = rng.normal(loc=5.0, scale=3.0, size=(1, 4, 6, 6))
print("eval mode, batch of 1: mean", np.round(bn.forward(xe).mean(axis=(0, 2, 3)), 2))
# a bias in front of batch norm is cancelled
w = rng.normal(size=(5, 4, 3, 3))
a1 = conv2d(xb, w, None, pad=1)
a2 = conv2d(xb, w, rng.normal(size=5), pad=1)
g, bt = rng.normal(size=5), rng.normal(size=5)
print("conv bias before BN has no effect:",
      np.allclose(batchnorm_forward(a1, g, bt)[0], batchnorm_forward(a2, g, bt)[0]))
c = 64
print(f"BN on {c} channels: {2 * c} trainable parameters + {2 * c} buffers (running mean and variance)")
print("Conv(3->64, 3x3)+BN:", params_grouped(3, 64, 3, bias=False) + 2 * 64, "trainable (bias=False)")

print("\n== 6. Batch of one, a single position (why training-mode BN needs m > 1)")
one = rng.normal(size=(1, 1, 1, 1))
print("normalized value:", batchnorm_forward(one, np.ones(1), np.zeros(1))[0].ravel())
