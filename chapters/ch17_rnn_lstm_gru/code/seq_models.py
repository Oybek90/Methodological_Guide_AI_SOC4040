# ci: skip
"""Training experiments of Chapter 17 (PyTorch, CPU, seeded, no download).

A. One-step-ahead prediction of a sensor-like series (the lecture used a CO2/temperature/humidity file that
   is not supplied; here a synthetic humidity-like series with 1008 samples replaces it).
B. A memory task: the label is the FIRST element of a random bit sequence of length T. Only a model that
   carries information over T-1 steps can solve it; the accuracy against T shows how far each cell remembers.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def make_series(n=1008, seed=0):
    """Humidity-like series: slow drift, a daily-type oscillation and small smooth noise (percent)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = np.convolve(rng.normal(0, 0.25, n + 20), np.ones(10) / 10, mode="same")[10:-10]
    return 40 - 0.004 * t + 5 * np.sin(2 * np.pi * t / 480) + 1.5 * np.sin(2 * np.pi * t / 97) + noise


def windows(x, k=3):
    X = np.stack([x[i:i + k] for i in range(len(x) - k)])
    return X, x[k:]


class SeqRegressor(nn.Module):
    def __init__(self, kind, hidden=4, d_in=1):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind](d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


def fit(model, X, y, epochs, lr, clip=1.0, loss_fn=nn.MSELoss()):
    opt = torch.optim.Adam(model.parameters(), lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X).squeeze(-1), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        hist.append(loss.item())
    return hist


def experiment_a(seed=0):
    x = make_series()
    x = (x - x.min()) / (x.max() - x.min())                      # min-max scaling as in the lecture notebook
    X, y = windows(x, 3)
    n = int(0.8 * len(X))                                         # time order: no shuffling
    Xtr, Xte, ytr, yte = X[:n], X[n:], y[:n], y[n:]
    res = {"Linear regression": mean_squared_error(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))}
    res["Last value (persistence)"] = mean_squared_error(yte, Xte[:, -1])
    Xtr_t, Xte_t = [torch.tensor(a, dtype=torch.float32).reshape(-1, 3, 1) for a in (Xtr, Xte)]
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    curves, preds = {}, {}
    for kind in ("RNN", "LSTM", "GRU"):
        torch.manual_seed(seed)
        m = SeqRegressor(kind, 4)
        curves[kind] = fit(m, Xtr_t, ytr_t, 1500, 0.02)
        with torch.no_grad():
            p = m(Xte_t).squeeze(-1).numpy()
        preds[kind] = p
        res[f"{kind} (4 units, {sum(q.numel() for q in m.parameters())} parameters)"] = mean_squared_error(yte, p)
    return res, curves, yte, preds


def memory_data(n, T, rng):
    X = rng.integers(0, 2, (n, T, 1)).astype(np.float32)
    return torch.tensor(X), torch.tensor(X[:, 0, 0])


def experiment_b(Ts=(10, 30, 60, 100), seeds=(0, 1, 2), hidden=32, epochs=500):
    acc = {k: {T: [] for T in Ts} for k in ("RNN", "LSTM", "GRU")}
    for T in Ts:
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            Xtr, ytr = memory_data(512, T, rng)
            Xte, yte = memory_data(1000, T, rng)
            for kind in acc:
                torch.manual_seed(seed)
                m = SeqRegressor(kind, hidden)
                fit(m, Xtr, ytr, epochs, 0.01, clip=1.0, loss_fn=nn.BCEWithLogitsLoss())
                with torch.no_grad():
                    acc[kind][T].append(float(((m(Xte).squeeze(-1) > 0).float() == yte).float().mean()))
    return acc


if __name__ == "__main__":
    t0 = time.time()
    res, curves, yte, preds = experiment_a()
    print("A. Test MSE (scaled to [0, 1]) for one-step-ahead prediction, window 3")
    for k, v in res.items():
        print(f"   {k:45s} {v:.2e}")
    np.savez("seq_a.npz", yte=yte, **{f"pred_{k}": v for k, v in preds.items()},
             **{f"loss_{k}": np.array(v) for k, v in curves.items()})
    print(f"   ({time.time() - t0:.0f} s)")
    t0 = time.time()
    acc = experiment_b()
    print("B. Memory task, test accuracy (mean +- std over 3 seeds), chance = 0.5")
    Ts = list(next(iter(acc.values())).keys())
    print("   T      " + "  ".join(f"{T:>11d}" for T in Ts))
    for k, d in acc.items():
        print(f"   {k:5s}  " + "  ".join(f"{np.mean(d[T]):.2f}+-{np.std(d[T]):.2f}" for T in Ts))
    np.savez("seq_b.npz", Ts=np.array(Ts), **{k: np.array([d[T] for T in Ts]) for k, d in acc.items()})
    print(f"   ({time.time() - t0:.0f} s)")
