# ci: run
"""Convolution and pooling layers in NumPy: forward, backward and a gradient check.

Tensors use the PyTorch layout (N, C, H, W); filters are (K, C, F, F).
Like every deep-learning library, "convolution" here means cross-correlation:
the filter is not flipped.
"""
import numpy as np


def out_size(n, f, s=1, p=0, strict=True):
    """Spatial output size (n - f + 2p)/s + 1. With strict=True a filter that does not fit raises;
    with strict=False the result is rounded down, as PyTorch does."""
    if strict and (n - f + 2 * p) % s != 0:
        raise ValueError(f"filter {f} with stride {s} does not fit input {n} (pad {p})")
    return (n - f + 2 * p) // s + 1


def conv_params(c_in, k, f, bias=True):
    """Parameters of a standard convolution: F*F*C*K weights plus K biases."""
    return f * f * c_in * k + (k if bias else 0)


def conv2d(x, w, b=None, stride=1, pad=0):
    n, c, h, wd = x.shape
    k, c2, f, _ = w.shape
    assert c == c2, "channel mismatch"
    ho, wo = out_size(h, f, stride, pad), out_size(wd, f, stride, pad)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    y = np.zeros((n, k, ho, wo))
    for i in range(ho):
        for j in range(wo):
            patch = xp[:, :, i * stride:i * stride + f, j * stride:j * stride + f]
            y[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return y + (0 if b is None else b.reshape(1, -1, 1, 1))


def conv2d_backward(dy, x, w, stride=1, pad=0):
    """Gradients of a loss with respect to x, w and b, given dy = dL/dy."""
    n, c, h, wd = x.shape
    k, _, f, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dxp, dw = np.zeros_like(xp), np.zeros_like(w)
    ho, wo = dy.shape[2:]
    for i in range(ho):
        for j in range(wo):
            sl = (slice(None), slice(None), slice(i * stride, i * stride + f),
                  slice(j * stride, j * stride + f))
            g = dy[:, :, i, j]                              # (N, K)
            dw += np.tensordot(g, xp[sl], axes=([0], [0]))  # (K, C, F, F)
            dxp[sl] += np.tensordot(g, w, axes=([1], [0]))  # (N, C, F, F)
    dx = dxp[:, :, pad:pad + h, pad:pad + wd]
    return dx, dw, dy.sum(axis=(0, 2, 3))


def pool2d(x, f=2, stride=2, mode="max"):
    n, c, h, wd = x.shape
    ho, wo = out_size(h, f, stride), out_size(wd, f, stride)
    y = np.zeros((n, c, ho, wo))
    for i in range(ho):
        for j in range(wo):
            win = x[:, :, i * stride:i * stride + f, j * stride:j * stride + f]
            y[:, :, i, j] = win.max(axis=(2, 3)) if mode == "max" else win.mean(axis=(2, 3))
    return y


def pool2d_backward(dy, x, f=2, stride=2, mode="max"):
    dx = np.zeros_like(x)
    ho, wo = dy.shape[2:]
    for i in range(ho):
        for j in range(wo):
            sl = (slice(None), slice(None), slice(i * stride, i * stride + f),
                  slice(j * stride, j * stride + f))
            if mode == "avg":
                dx[sl] += dy[:, :, i, j][:, :, None, None] / (f * f)
            else:
                win = x[sl]
                m = (win == win.max(axis=(2, 3), keepdims=True))
                m = m / m.sum(axis=(2, 3), keepdims=True)   # split ties equally
                dx[sl] += m * dy[:, :, i, j][:, :, None, None]
    return dx


def numeric_grad(fun, a, eps=1e-6):
    g = np.zeros_like(a)
    it = np.nditer(a, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = a[idx]
        a[idx] = old + eps
        fp = fun()
        a[idx] = old - eps
        fm = fun()
        a[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def rel_err(a, b):
    return np.abs(a - b).max() / max(1e-12, np.abs(a).max() + np.abs(b).max())


def gradient_check(seed=0, stride=2, pad=1):
    """conv -> max-pool -> sum of squares; compare analytic and numeric gradients."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, 3, 7, 7))
    w = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=4)

    def loss():
        y = conv2d(x, w, b, stride, pad)
        return 0.5 * np.sum(pool2d(y, 2, 1, "max") ** 2)

    y = conv2d(x, w, b, stride, pad)
    p = pool2d(y, 2, 1, "max")
    dy = pool2d_backward(p, y, 2, 1, "max")
    dx, dw, db = conv2d_backward(dy, x, w, stride, pad)
    return {"dx": rel_err(dx, numeric_grad(loss, x)), "dw": rel_err(dw, numeric_grad(loss, w)),
            "db": rel_err(db, numeric_grad(loss, b))}


if __name__ == "__main__":
    errs = gradient_check()
    print("relative errors:", {k: f"{v:.1e}" for k, v in errs.items()})
    assert max(errs.values()) < 1e-6
    x = np.random.default_rng(1).normal(size=(1, 1, 6, 6))
    ya = pool2d(x, 2, 2, "avg")
    dxa = pool2d_backward(np.ones_like(ya), x, 2, 2, "avg")
    assert np.allclose(dxa, 0.25)
    print("cnn_numpy self-test passed")
