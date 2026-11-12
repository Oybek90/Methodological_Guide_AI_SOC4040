# ci: run
"""Reproduces every number of Chapter 14 that does not need PyTorch."""
import numpy as np
from sklearn.datasets import load_digits
from cnn_numpy import out_size, conv_params, conv2d, pool2d

np.set_printoptions(linewidth=120)

# 1. The 7x7 example of slides 50-54 (zero padding 1, stride 1, no bias) -------------------
X = np.array([[2, 1, 1, 2, 4, 2, 1], [3, 2, 3, 2, 3, 1, 1], [1, 1, 2, 2, 2, 3, 3],
              [1, 2, 1, 3, 1, 2, 1], [2, 2, 2, 2, 2, 0, 2], [3, 2, 3, 1, 3, 2, 1],
              [2, 3, 2, 3, 1, 2, 1]], dtype=float)
K = np.array([[1, 2, 1], [0, 0, 0], [1, 2, 1]], dtype=float)
Y = conv2d(X[None, None], K[None, None], pad=1)[0, 0]
SLIDE = np.array([[8, 10, 10, 10, 9, 6, 3], [6, 12, 12, 17, 21, 20, 13], [11, 16, 17, 17, 16, 12, 7],
                  [7, 13, 15, 16, 15, 15, 13], [12, 16, 16, 16, 16, 14, 8], [13, 18, 20, 17, 13, 11, 8],
                  [8, 10, 9, 8, 9, 8, 4]])
print("output of the slide-54 example:\n", Y.astype(int))
bad = np.argwhere(Y != SLIDE)
print("cells that differ from slide 54 (row, col, computed, slide):",
      [(int(i) + 1, int(j) + 1, int(Y[i, j]), int(SLIDE[i, j])) for i, j in bad])
print("number of differing cells:", len(bad), "of 49")
Yv = conv2d(X[None, None], K[None, None])[0, 0]
print("valid (no padding) output shape:", Yv.shape)

# 2. Output sizes (slides 38-59) ----------------------------------------------------------
print("7x7, F=3, S=1:", out_size(7, 3, 1), "| S=2:", out_size(7, 3, 2), "| P=1:", out_size(7, 3, 1, 1))
try:
    out_size(7, 3, 3)
except ValueError as e:
    print("S=3:", e)
print("32x32x3, ten 5x5 filters, S=1, P=2:", out_size(32, 5, 1, 2), "params:", conv_params(3, 10, 5))
print("same padding P=(F-1)/2 keeps size:", [out_size(28, f, 1, (f - 1) // 2) for f in (3, 5, 7)])

# 3. Pooling (slides 62-64) ----------------------------------------------------------------
Pm = np.array([[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]], dtype=float)[None, None]
print("max pool:\n", pool2d(Pm, 2, 2, "max")[0, 0], "\navg pool:\n", pool2d(Pm, 2, 2, "avg")[0, 0])

# 4. Why convolution: parameters ----------------------------------------------------------
fc = (32 * 32 * 3 + 1) * (32 * 32 * 10)
print("fully connected 32x32x3 -> 32x32x10:", fc, "| convolution:", conv_params(3, 10, 5),
      "| ratio %.0f" % (fc / conv_params(3, 10, 5)))
print("notebook layers (3->64): 3x3:", conv_params(3, 64, 3), "5x5:", conv_params(3, 64, 5),
      "7x7:", conv_params(3, 64, 7), "| 3x3+7x7:", conv_params(3, 64, 3) + conv_params(3, 64, 7))
std = conv_params(32, 64, 3)
grp = 64 * 1 * 9 + 64                       # groups = 32: each filter sees 32/32 = 1 channel
sep = (32 * 9 + 32) + (32 * 64 + 64)        # depthwise 3x3 + pointwise 1x1
print("32->64, 3x3: standard", std, "grouped(32)", grp, "separable", sep, "| ratio %.1f" % (std / sep))
for c in (16, 64, 256):
    print(f"C={c}: three 3x3 layers {3 * 9 * c * c} weights vs one 7x7 layer {49 * c * c}")

# 5. Receptive field of stacked 3x3 convolutions -------------------------------------------
rf, jump = 1, 1
for layer in range(1, 5):
    rf += 2 * jump
    print(f"after {layer} conv(3x3, S=1): receptive field {rf}x{rf}")

# 6. Equivariance and the effect of pooling ------------------------------------------------
img = load_digits().images[3] / 16.0
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
a = conv2d(img[None, None], sobel[None, None], pad=1)[0, 0]
shifted = np.roll(img, 1, axis=1)
b = conv2d(shifted[None, None], sobel[None, None], pad=1)[0, 0]
print("equivariance (interior columns), max |shift(conv(x)) - conv(shift(x))|:",
      float(np.abs(np.roll(a, 1, axis=1)[:, 2:-2] - b[:, 2:-2]).max()))
pa = pool2d(np.abs(a)[None, None], 2, 2, "max")[0, 0]
pb = pool2d(np.abs(b)[None, None], 2, 2, "max")[0, 0]
print("mean |difference| after a 1-pixel shift: pooled %.3f, unpooled %.3f"
      % (np.abs(pa - pb).mean(), np.abs(np.abs(a) - np.abs(b)).mean()))

# 7. Parameter counts of the notebook architectures -----------------------------------------
lenet_nb = conv_params(1, 6, 5) + conv_params(6, 16, 3) + (256 * 120 + 120) + (120 * 84 + 84) + (84 * 10 + 10)
lenet_cl = conv_params(1, 6, 5) + conv_params(6, 16, 5) + (256 * 120 + 120) + (120 * 84 + 84) + (84 * 10 + 10)
print("LeNet-5 as in the notebook (conv2 3x3, pool 5/2):", lenet_nb, "| classic (conv2 5x5, pool 2/2):", lenet_cl)
print("notebook sizes 28 ->", out_size(28, 5), "-> pool(5,2)", out_size(24, 5, 2, strict=False), "(rounded down: 9.5+1)", "-> conv3", out_size(10, 3), "-> pool2", out_size(8, 2, 2))
print("classic sizes 28 ->", out_size(28, 5), "-> pool", out_size(24, 2, 2), "-> conv5", out_size(12, 5), "-> pool", out_size(8, 2, 2))
fc_alex = (256 * 6 * 6 * 4096 + 4096) + (4096 * 4096 + 4096) + (4096 * 10 + 10)
conv_alex = conv_params(1, 96, 11) + conv_params(96, 256, 5) + conv_params(256, 384, 3)
alex_nb = conv_alex + conv_params(384, 256, 3) + fc_alex                              # notebook: 4 conv layers
alex_ok = conv_alex + conv_params(384, 384, 3) + conv_params(384, 256, 3) + fc_alex  # original: 5 conv layers
print("AlexNet variant, 1 input channel, 10 classes: notebook (4 conv)", alex_nb, "| with the missing conv5", alex_ok,
      "| share of the three dense layers: %.1f%%" % (100 * fc_alex / alex_ok))
n = 224
for f, s, p in [(11, 4, 2), (3, 2, 0), (5, 1, 2), (3, 2, 0), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 2, 0)]:
    n = out_size(n, f, s, p, strict=False)
print("224x224 input through the AlexNet stack ends at", n, "x", n, "-> 256*6*6 =", 256 * 6 * 6)
print("28x28 input: conv1", out_size(28, 11, 4, 2, strict=False), "-> pool1", out_size(6, 3, 2, strict=False),
      "-> conv2", out_size(2, 5, 1, 2), "-> pool2", out_size(2, 3, 2, strict=False), "(the image vanishes)")
