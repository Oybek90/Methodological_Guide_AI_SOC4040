# ci: run
"""Numbers of Chapter 17: exploding and vanishing gradients, one hand-checked BPTT step, parameter counts."""
import numpy as np
from rnn_np import (init_rnn, rnn_forward, rnn_loss_grads, init_lstm, lstm_forward, lstm_loss_grads,
                    n_params, clip_by_norm, gradient_check)

print("== 1. Slides 29-45: the same weight multiplied T times")
for w in (2.0, 0.5, 1.0):
    print(f"w = {w}: w^3 = {w ** 3:.4g}   w^50 = {w ** 50:.4g}")

print("\n== 2. Scalar RNN, h_t = tanh(w h_{t-1} + x_t): factor d h_T / d h_1 after T-1 steps")
rng = np.random.default_rng(0)
xs = rng.normal(0, 0.5, 51)
for w in (0.5, 1.0, 2.0):
    h, jac, out = 0.0, 1.0, {}
    for t in range(1, 51):
        h = np.tanh(w * h + xs[t])
        if t > 1:
            jac *= w * (1 - h ** 2)
        if t in (5, 20, 50):
            out[t] = jac
    print(f"w = {w}:", {k: f"{v:.2e}" for k, v in out.items()})

print("\n== 3. Vector RNN (H = 32): norm of d h_T / d h_t for growing lag, T = 50")
H, T = 32, 50
lags = (1, 5, 10, 20, 49)
for rho in (0.5, 0.9, 1.5):
    rng = np.random.default_rng(1)
    Wh = rng.normal(size=(H, H)); Wh *= rho / max(abs(np.linalg.eigvals(Wh)))     # spectral radius rho
    p = {"Wx": rng.normal(0, 0.5, (H, 1)), "Wh": Wh, "b": np.zeros(H), "Wy": np.zeros(H), "by": 0.0}
    hs, _ = rnn_forward(rng.normal(size=(T, 1)), p)
    M, norms = np.eye(H), {}
    for k in range(T, 0, -1):
        if T - k in lags:
            norms[T - k] = np.linalg.norm(M, 2)
        M = M @ (np.diag(1 - hs[k] ** 2) @ Wh)
    print(f"spectral radius {rho}:", {k: f"{v:.1e}" for k, v in norms.items()})

print("\n== 4. LSTM cell state: d c_T / d c_t = product of the forget gates (constant gate f)")
for f in (0.5, 0.9, 0.99, 1.0):
    print(f"f = {f}: f^10 = {f ** 10:.3g}   f^50 = {f ** 50:.3g}")

print("\n== 5. One BPTT step by hand (RNN_practice.ipynb): x = (0, 0.01323155, 0.02264631), y = 0.03231552")
x = np.array([0.0, 0.01323155, 0.02264631]); y = 0.03231552
wx, wh, wy = 0.3745401188, 0.9507143064, 0.7319939418            # np.random.seed(42); np.random.rand(3)
p = {"Wx": np.array([[wx]]), "Wh": np.array([[wh]]), "b": np.zeros(1), "Wy": np.array([wy]), "by": 0.0}
hs, yhat = rnn_forward(x.reshape(-1, 1), p)
print("h1, h2, h3 =", np.round([h[0] for h in hs[1:]], 6), " y_hat = %.6f" % yhat, " loss = %.3e" % (0.5 * (yhat - y) ** 2))
L, g = rnn_loss_grads(x.reshape(-1, 1), y, p)
print("dL/dwy = %.6f  dL/dby = %.6f  dL/dwx = %.6f  dL/dwh = %.6f  dL/dbh = %.6f"
      % (g["Wy"][0], g["by"], g["Wx"][0, 0], g["Wh"][0, 0], g["b"][0]))
print("one step of SGD with lr 0.01 lowers the loss:", end=" ")
q = {k: (v - 0.01 * g[k]) for k, v in p.items()}
_, yhat2 = rnn_forward(x.reshape(-1, 1), q)
print("%.3e -> %.3e" % (0.5 * (yhat - y) ** 2, 0.5 * (yhat2 - y) ** 2))

print("\n== 6. Gradient check by finite differences (largest absolute error)")
print("RNN : %.1e   LSTM: %.1e" % (gradient_check("rnn"), gradient_check("lstm")))

print("\n== 7. Parameters (one bias vector per gate), input d, hidden H")
for d, H in ((1, 1), (1, 4), (1, 5), (10, 64), (128, 64)):
    print(f"d={d:3d} H={H:3d}: RNN {n_params('rnn', d, H):6d}  GRU {n_params('gru', d, H):6d}  LSTM {n_params('lstm', d, H):6d}")

print("\n== 8. Gradient clipping")
gr = {"a": np.array([30.0, 40.0])}
c, tot = clip_by_norm(gr, 5.0)
print("norm before", tot, "after", float(np.linalg.norm(c["a"])), c["a"])

print("\n== 9. One LSTM step by hand: H = 1, every gate has [w_h, w_x] = [0.5, 1.0], b_f = 1, other biases 0")
from rnn_np import lstm_step
W = np.array([[0.5, 1.0]] * 4); b = np.array([0.0, 1.0, 0.0, 0.0])      # rows i, f, o, g
h, c = np.zeros(1), np.zeros(1)
for t, x in enumerate((1.0, 0.0), 1):
    h, c, (i, f, o, g) = lstm_step(np.array([x]), h, c, W, b)
    print(f"t={t}, x={x}: i={i[0]:.4f} f={f[0]:.4f} o={o[0]:.4f} g={g[0]:.4f}  c={c[0]:.4f}  h={h[0]:.4f}")

print("\n== 10. Exercises")
print("1.1^100 = %.1f   0.9^100 = %.3e   w with w^50 = 1e6: %.3f" % (1.1 ** 100, 0.9 ** 100, 10 ** (6 / 50)))
p = {"Wx": np.array([[1.0]]), "Wh": np.array([[0.5]]), "b": np.zeros(1), "Wy": np.array([1.0]), "by": 0.0}
xx = np.array([[1.0], [0.0]]); hs, yh = rnn_forward(xx, p); L, gr = rnn_loss_grads(xx, 0.5, p)
print("T=2 scalar RNN: h1=%.5f h2=%.5f y_hat=%.5f loss=%.5f" % (hs[1][0], hs[2][0], yh, L),
      " dWx=%.5f dWh=%.5f db=%.5f dWy=%.5f" % (gr["Wx"][0, 0], gr["Wh"][0, 0], gr["b"][0], gr["Wy"][0]))
