# ci: run
"""Recurrent layers in NumPy (Chapter 17): vanilla RNN, LSTM and GRU.

Every model is many-to-one: it reads a sequence X of shape (T, d) and predicts one number
y_hat = Wy . h_T + by, with the loss L = 0.5 (y_hat - y)^2. Gradients are computed by
backpropagation through time (BPTT) and verified by finite differences in the self-test.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---- vanilla RNN:  h_t = tanh(Wx x_t + Wh h_{t-1} + b) ----
def init_rnn(d, H, rng, scale=0.1):
    return {"Wx": rng.normal(0, scale, (H, d)), "Wh": rng.normal(0, scale, (H, H)),
            "b": np.zeros(H), "Wy": rng.normal(0, scale, H), "by": 0.0}


def rnn_forward(X, p, h0=None):
    """Return the list of hidden states h_0, ..., h_T and the prediction."""
    H = p["Wh"].shape[0]
    hs = [np.zeros(H) if h0 is None else h0]
    for x in X:
        hs.append(np.tanh(p["Wx"] @ x + p["Wh"] @ hs[-1] + p["b"]))
    return hs, float(p["Wy"] @ hs[-1] + p["by"])


def rnn_loss_grads(X, y, p):
    hs, yhat = rnn_forward(X, p)
    err = yhat - y
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["Wy"], g["by"] = err * hs[-1], err
    dh = err * p["Wy"]
    for t in range(len(X), 0, -1):                 # backwards in time
        dz = dh * (1 - hs[t] ** 2)                 # through tanh
        g["Wx"] += np.outer(dz, X[t - 1])
        g["Wh"] += np.outer(dz, hs[t - 1])
        g["b"] += dz
        dh = p["Wh"].T @ dz                        # to the previous time step
    return 0.5 * err ** 2, g


# ---- LSTM (slide 50): [i f o g] = (sigma sigma sigma tanh) W [h_{t-1}; x_t] ----
def init_lstm(d, H, rng, scale=0.1):
    return {"W": rng.normal(0, scale, (4 * H, H + d)), "b": np.zeros(4 * H),
            "Wy": rng.normal(0, scale, H), "by": 0.0}


def lstm_step(x, h, c, W, b):
    H = len(h)
    z = W @ np.concatenate([h, x]) + b
    i, f, o = sigmoid(z[:H]), sigmoid(z[H:2 * H]), sigmoid(z[2 * H:3 * H])
    g = np.tanh(z[3 * H:])
    c_new = f * c + i * g
    return o * np.tanh(c_new), c_new, (i, f, o, g)


def lstm_forward(X, p):
    H = len(p["b"]) // 4
    h, c = np.zeros(H), np.zeros(H)
    cache = []
    for x in X:
        h_new, c_new, gates = lstm_step(x, h, c, p["W"], p["b"])
        cache.append((x, h, c, c_new, gates))
        h, c = h_new, c_new
    return h, c, cache, float(p["Wy"] @ h + p["by"])


def lstm_loss_grads(X, y, p):
    h, c, cache, yhat = lstm_forward(X, p)
    H = len(h)
    err = yhat - y
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["Wy"], g["by"] = err * h, err
    dh, dc = err * p["Wy"], np.zeros(H)
    for x, h_prev, c_prev, c_t, (i, f, o, gg) in reversed(cache):
        tc = np.tanh(c_t)
        do = dh * tc
        dc = dc + dh * o * (1 - tc ** 2)           # the cell state receives gradient from h and from t+1
        di, dgg, df = dc * gg, dc * i, dc * c_prev
        dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dgg * (1 - gg ** 2)])
        inp = np.concatenate([h_prev, x])
        g["W"] += np.outer(dz, inp)
        g["b"] += dz
        dh = (p["W"].T @ dz)[:H]
        dc = dc * f                                # d c_t / d c_{t-1} = f: the "uninterrupted" path
    return 0.5 * err ** 2, g


# ---- GRU (slide 62): z, r gates; h_t = (1 - z) h_{t-1} + z h_hat ----
def init_gru(d, H, rng, scale=0.1):
    return {"Wz": rng.normal(0, scale, (H, d)), "Uz": rng.normal(0, scale, (H, H)), "bz": np.zeros(H),
            "Wr": rng.normal(0, scale, (H, d)), "Ur": rng.normal(0, scale, (H, H)), "br": np.zeros(H),
            "Wh": rng.normal(0, scale, (H, d)), "Uh": rng.normal(0, scale, (H, H)), "bh": np.zeros(H)}


def gru_step(x, h, p):
    z = sigmoid(p["Wz"] @ x + p["Uz"] @ h + p["bz"])               # update gate
    r = sigmoid(p["Wr"] @ x + p["Ur"] @ h + p["br"])               # reset gate
    hh = np.tanh(p["Wh"] @ x + p["Uh"] @ (r * h) + p["bh"])        # candidate state
    return (1 - z) * h + z * hh


def gru_forward(X, p):
    h = np.zeros(len(p["bz"]))
    for x in X:
        h = gru_step(x, h, p)
    return h


def n_params(kind, d, H):
    """Number of parameters with one bias vector per gate (the convention of the equations)."""
    return {"rnn": 1, "gru": 3, "lstm": 4}[kind] * (H * (H + d) + H)


def clip_by_norm(grads, max_norm):
    """Gradient clipping: rescale all gradients together if their global norm exceeds max_norm."""
    total = np.sqrt(sum(float(np.sum(np.square(v))) for v in grads.values()))
    s = min(1.0, max_norm / (total + 1e-12))
    return {k: v * s for k, v in grads.items()}, total


def numeric_grad(loss_fn, p, key, eps=1e-6):
    """Central finite differences for the parameter array p[key]."""
    a = np.atleast_1d(np.asarray(p[key], dtype=float)).copy()
    out = np.zeros_like(a)
    for idx in np.ndindex(*a.shape):
        old = a[idx]
        a[idx] = old + eps; p[key] = a.copy() if np.ndim(p[key]) else float(a[0]); lp = loss_fn()
        a[idx] = old - eps; p[key] = a.copy() if np.ndim(p[key]) else float(a[0]); lm = loss_fn()
        a[idx] = old; p[key] = a.copy() if np.ndim(p[key]) else float(a[0])
        out[idx] = (lp - lm) / (2 * eps)
    return out


def gradient_check(kind, seed=0, T=5, d=2, H=3):
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(T, d)), 0.7
    p = (init_rnn if kind == "rnn" else init_lstm)(d, H, rng, 0.5)
    fwd = rnn_forward if kind == "rnn" else lstm_forward
    grads = (rnn_loss_grads if kind == "rnn" else lstm_loss_grads)(X, y, p)[1]
    loss_fn = lambda: 0.5 * (fwd(X, p)[-1] - y) ** 2
    worst = 0.0
    for k in p:
        num = numeric_grad(loss_fn, p, k)
        worst = max(worst, float(np.max(np.abs(num - np.atleast_1d(grads[k])))))
    return worst


# ---- self-test ----
if __name__ == "__main__":
    assert gradient_check("rnn") < 1e-7, gradient_check("rnn")
    assert gradient_check("lstm") < 1e-7, gradient_check("lstm")
    rng = np.random.default_rng(1)
    pg = init_gru(2, 3, rng)
    h = gru_forward(rng.normal(size=(4, 2)), pg)
    assert h.shape == (3,) and np.all(np.abs(h) < 1)
    assert n_params("lstm", 1, 1) == 12 and n_params("gru", 1, 1) == 9 and n_params("rnn", 1, 1) == 3
    g, tot = clip_by_norm({"a": np.array([3.0, 4.0])}, 1.0)
    assert abs(tot - 5) < 1e-12 and abs(np.linalg.norm(g["a"]) - 1) < 1e-9
    print("rnn_np self-test passed")
