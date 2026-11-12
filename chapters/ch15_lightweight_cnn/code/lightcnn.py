# ci: run
"""Lightweight convolutions and batch normalization in NumPy.

Builds on the standard convolution of Chapter 14 (cnn_numpy.py, copied next to this file).
Tensors use the layout (N, C, H, W); a grouped filter bank has shape (K, C/g, F, F).
"""
import numpy as np
from cnn_numpy import conv2d, conv2d_backward, out_size, numeric_grad, rel_err


# ---- cost formulas ----
def params_grouped(c_in, k, f, groups=1, bias=True):
    """Parameters of a grouped convolution: K*(C/g)*F*F weights (+K biases)."""
    assert c_in % groups == 0 and k % groups == 0, "groups must divide both C and K"
    return k * (c_in // groups) * f * f + (k if bias else 0)


def macs_grouped(c_in, k, f, h_out, w_out, groups=1):
    """Multiply-adds of a grouped convolution (biases not counted)."""
    return k * (c_in // groups) * f * f * h_out * w_out


def params_separable(c_in, k, f, bias=True):
    """Depthwise (C filters F x F) followed by pointwise (K filters 1 x 1 x C)."""
    return c_in * f * f + k * c_in + ((c_in + k) if bias else 0)


def macs_separable(c_in, k, f, h_out, w_out):
    return (c_in * f * f + k * c_in) * h_out * w_out


# ---- grouped, depthwise and pointwise convolution ----
def grouped_conv2d(x, w, b=None, groups=1, stride=1, pad=0):
    """Split the input channels into `groups` blocks; block i is convolved only with the i-th
    block of K/groups filters, and the outputs are concatenated."""
    n, c, _, _ = x.shape
    k, cg, _, _ = w.shape
    assert c == cg * groups and k % groups == 0, "shape mismatch"
    kg = k // groups
    outs = [conv2d(x[:, g * cg:(g + 1) * cg], w[g * kg:(g + 1) * kg], None, stride, pad)
            for g in range(groups)]
    y = np.concatenate(outs, axis=1)
    return y + (0 if b is None else b.reshape(1, -1, 1, 1))


def depthwise_conv2d(x, w, b=None, stride=1, pad=0):
    """Depthwise convolution = grouped convolution with groups = C. w has shape (C, 1, F, F)."""
    return grouped_conv2d(x, w, b, groups=x.shape[1], stride=stride, pad=pad)


def pointwise_conv2d(x, w, b=None):
    """1 x 1 convolution: mixes the channels at every pixel. w has shape (K, C, 1, 1)."""
    return conv2d(x, w, b)


def separable_conv2d(x, w_dw, w_pw, stride=1, pad=0):
    return pointwise_conv2d(depthwise_conv2d(x, w_dw, None, stride, pad), w_pw)


def grouped_conv2d_backward(dy, x, w, groups=1, stride=1, pad=0):
    cg, kg = w.shape[1], w.shape[0] // groups
    dx, dw = np.zeros_like(x), np.zeros_like(w)
    for g in range(groups):
        dxg, dwg, _ = conv2d_backward(dy[:, g * kg:(g + 1) * kg], x[:, g * cg:(g + 1) * cg],
                                      w[g * kg:(g + 1) * kg], stride, pad)
        dx[:, g * cg:(g + 1) * cg], dw[g * kg:(g + 1) * kg] = dxg, dwg
    return dx, dw, dy.sum(axis=(0, 2, 3))


# ---- batch normalization for feature maps ----
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """Normalize every channel over the batch and both spatial axes (N, H, W)."""
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)          # biased: divides by N*H*W
    xhat = (x - mu) / np.sqrt(var + eps)
    y = gamma.reshape(1, -1, 1, 1) * xhat + beta.reshape(1, -1, 1, 1)
    return y, (xhat, var, gamma, eps), mu.ravel(), var.ravel()


def batchnorm_backward(dy, cache):
    xhat, var, gamma, eps = cache
    m = xhat.shape[0] * xhat.shape[2] * xhat.shape[3]
    g = gamma.reshape(1, -1, 1, 1)
    dgamma, dbeta = (dy * xhat).sum(axis=(0, 2, 3)), dy.sum(axis=(0, 2, 3))
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(axis=(0, 2, 3), keepdims=True)
          - xhat * (dxhat * xhat).mean(axis=(0, 2, 3), keepdims=True)) / np.sqrt(var + eps)
    return dx, dgamma, dbeta


class BatchNorm2d:
    """Training mode uses batch statistics and updates running averages;
    evaluation mode uses the running averages."""

    def __init__(self, channels, eps=1e-5, momentum=0.1):
        self.gamma, self.beta = np.ones(channels), np.zeros(channels)
        self.run_mean, self.run_var = np.zeros(channels), np.ones(channels)
        self.eps, self.momentum, self.training = eps, momentum, True

    def forward(self, x):
        if self.training:
            y, self.cache, mu, var = batchnorm_forward(x, self.gamma, self.beta, self.eps)
            m = x.shape[0] * x.shape[2] * x.shape[3]
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * mu
            unbiased = var * m / (m - 1)
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * unbiased
            return y
        r = lambda a: a.reshape(1, -1, 1, 1)
        xh = (x - r(self.run_mean)) / np.sqrt(r(self.run_var) + self.eps)
        return r(self.gamma) * xh + r(self.beta)


# ---- self-test ----
def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, 4, 5, 5))
    w = rng.normal(size=(6, 2, 3, 3))                    # groups=2: 2 channels per group
    r = rng.normal(size=grouped_conv2d(x, w, groups=2, pad=1).shape)
    loss = lambda: float((grouped_conv2d(x, w, groups=2, pad=1) * r).sum())
    dx, dw, _ = grouped_conv2d_backward(r, x, w, groups=2, pad=1)
    errs = {"grouped dx": rel_err(dx, numeric_grad(loss, x)),
            "grouped dw": rel_err(dw, numeric_grad(loss, w))}
    gam, bet = rng.normal(size=4) + 2, rng.normal(size=4)
    rr = rng.normal(size=x.shape)
    loss = lambda: float((batchnorm_forward(x, gam, bet)[0] * rr).sum())
    _, cache, _, _ = batchnorm_forward(x, gam, bet)
    dx, dg, db = batchnorm_backward(rr, cache)
    errs["bn dx"] = rel_err(dx, numeric_grad(loss, x))
    errs["bn dgamma"] = rel_err(dg, numeric_grad(loss, gam))
    errs["bn dbeta"] = rel_err(db, numeric_grad(loss, bet))
    return errs


if __name__ == "__main__":
    for name, e in gradient_check().items():
        print(f"{name:11s} relative error {e:.1e}")
        assert e < 1e-6
    # groups=1 equals the standard convolution; groups=C equals a per-channel loop
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(2, 3, 6, 6)), rng.normal(size=(4, 3, 3, 3))
    assert np.allclose(grouped_conv2d(x, w, groups=1), conv2d(x, w))
    wd = rng.normal(size=(3, 1, 3, 3))
    ref = np.concatenate([conv2d(x[:, [c]], wd[[c]]) for c in range(3)], axis=1)
    assert np.allclose(depthwise_conv2d(x, wd), ref)
    print("self-test passed")
