# ci: skip
"""PyTorch recurrent layers (Chapter 17): equivalence with the NumPy cells, shapes and parameter counts."""
import numpy as np
import torch
import torch.nn as nn
from rnn_np import init_rnn, rnn_forward, init_lstm, lstm_forward, sigmoid

torch.manual_seed(0)
rng = np.random.default_rng(0)
d, H, T = 3, 4, 6
X = rng.normal(size=(T, d))
Xt = torch.tensor(X, dtype=torch.float64).reshape(1, T, d)

# 1. nn.RNN equals our NumPy RNN (PyTorch has two bias vectors, b_ih + b_hh)
p = init_rnn(d, H, rng, 0.5)
m = nn.RNN(d, H, batch_first=True).double()
with torch.no_grad():
    m.weight_ih_l0.copy_(torch.tensor(p["Wx"])); m.weight_hh_l0.copy_(torch.tensor(p["Wh"]))
    m.bias_ih_l0.copy_(torch.tensor(p["b"])); m.bias_hh_l0.zero_()
out, h_n = m(Xt)
print("nn.RNN  vs NumPy, max |difference| = %.1e" % np.max(np.abs(out[0, -1].detach().numpy() - rnn_forward(X, p)[0][-1])))

# 2. nn.LSTM equals our NumPy LSTM; PyTorch orders the gates i, f, g, o, the slides i, f, o, g
q = init_lstm(d, H, rng, 0.5)
m = nn.LSTM(d, H, batch_first=True).double()
order = np.r_[0:H, H:2 * H, 3 * H:4 * H, 2 * H:3 * H]                    # i f o g  ->  i f g o
W = q["W"][order]
with torch.no_grad():
    m.weight_hh_l0.copy_(torch.tensor(W[:, :H])); m.weight_ih_l0.copy_(torch.tensor(W[:, H:]))
    m.bias_ih_l0.copy_(torch.tensor(q["b"][order])); m.bias_hh_l0.zero_()
out, (h_n, c_n) = m(Xt)
h_np, c_np, _, _ = lstm_forward(X, q)
print("nn.LSTM vs NumPy, max |difference| h: %.1e, c: %.1e" % (np.max(np.abs(h_n[0, 0].detach().numpy() - h_np)),
                                                              np.max(np.abs(c_n[0, 0].detach().numpy() - c_np))))

# 3. nn.GRU: PyTorch writes h = (1 - z) n + z h_prev (z has the opposite meaning to slide 62) and
#    n = tanh(W_in x + b_in + r * (W_hn h + b_hn)); we reproduce it by hand
g = nn.GRU(d, H, batch_first=True).double()
Wi, Wh_, bi, bh = [t.detach().numpy() for t in (g.weight_ih_l0, g.weight_hh_l0, g.bias_ih_l0, g.bias_hh_l0)]
h = np.zeros(H)
for x in X:
    r = sigmoid(Wi[:H] @ x + bi[:H] + Wh_[:H] @ h + bh[:H])
    z = sigmoid(Wi[H:2 * H] @ x + bi[H:2 * H] + Wh_[H:2 * H] @ h + bh[H:2 * H])
    n = np.tanh(Wi[2 * H:] @ x + bi[2 * H:] + r * (Wh_[2 * H:] @ h + bh[2 * H:]))
    h = (1 - z) * n + z * h
print("nn.GRU  vs hand-written (PyTorch form), max |difference| = %.1e" % np.max(np.abs(g(Xt)[0][0, -1].detach().numpy() - h)))

# 4. parameter counts of the models of the lecture notebooks (PyTorch: two bias vectors per gate)
count = lambda mod: sum(t.numel() for t in mod.parameters())
for hid in (1, 4, 5, 64):
    print(f"hidden {hid:2d}, input 1: nn.RNN {count(nn.RNN(1, hid)):6d}  nn.GRU {count(nn.GRU(1, hid)):6d}  nn.LSTM {count(nn.LSTM(1, hid)):6d}")

# 5. shapes and the batch_first pitfall
x = torch.rand(1004, 3, 1)                      # (samples, time steps, features)
o, _ = nn.LSTM(1, 5)(x)                         # default batch_first=False: reads dimension 0 as TIME
print("batch_first=False: out shape", tuple(o.shape), "-> the 1004 samples are read as ONE sequence of 1004 steps with batch 3;\n  out[:, -1] is then the last of 3 \"sequences\", not the last time step")
o, (hn, cn) = nn.LSTM(1, 5, batch_first=True)(x)
print("batch_first=True : out shape", tuple(o.shape), " h_n", tuple(hn.shape), " c_n", tuple(cn.shape),
      " last step equal to h_n:", bool(torch.allclose(o[:, -1], hn[0])))
