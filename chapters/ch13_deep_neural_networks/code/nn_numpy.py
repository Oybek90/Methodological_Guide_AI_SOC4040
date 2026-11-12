# ci: run
"""A small fully connected network in NumPy: activations, losses, backpropagation, gradient check."""
import numpy as np

# ---------- numerically safe building blocks ----------


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])                      # exp of a negative number: never overflows
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z):
    """Row-wise softmax; subtracting the row maximum does not change the result but avoids overflow."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


ALPHA_SELU, SCALE_SELU = 1.6732632423543772, 1.0507009873554805


def _selu(z):
    return SCALE_SELU * np.where(z > 0, z, ALPHA_SELU * np.expm1(np.minimum(z, 0)))


# name -> (g(z), g'(z)); every function takes and returns arrays
ACT = {
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.01 * z), lambda z: np.where(z > 0, 1.0, 0.01)),
    "elu": (
        lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0))),
        lambda z: np.where(z > 0, 1.0, np.exp(np.minimum(z, 0))),
    ),
    "selu": (_selu, lambda z: SCALE_SELU * np.where(z > 0, 1.0, ALPHA_SELU * np.exp(np.minimum(z, 0)))),
    "swish": (lambda z: z * sigmoid(z), lambda z: sigmoid(z) * (1 + z * (1 - sigmoid(z)))),
    "identity": (lambda z: z, lambda z: np.ones_like(z)),
}


class MLP:
    """Layers `sizes`; hidden activation `act`; output 'linear' (MSE), 'sigmoid' (BCE) or 'softmax' (CE)."""

    def __init__(self, sizes, act="relu", out="linear", init="he", seed=0):
        self.sizes, self.act, self.out = list(sizes), act, out
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            if init == "he":
                s = np.sqrt(2.0 / n_in)
            elif init == "xavier":
                s = np.sqrt(2.0 / (n_in + n_out))
            elif init == "lecun":
                s = np.sqrt(1.0 / n_in)
            else:
                s = float(init)                # a number: fixed standard deviation
            self.W.append(rng.normal(0.0, s, (n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

    # ----- forward -----
    def forward(self, X):
        g = ACT[self.act][0]
        A, Z = [X], []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            Z.append(z)
            if l < len(self.W) - 1:
                A.append(g(z))
            elif self.out == "sigmoid":
                A.append(sigmoid(z))
            elif self.out == "softmax":
                A.append(softmax(z))
            else:
                A.append(z)
        return A, Z

    def predict(self, X):
        return self.forward(X)[0][-1]

    def loss(self, X, Y):
        P = self.predict(X)
        if self.out == "linear":
            return np.mean(np.sum((P - Y) ** 2, axis=1))
        if self.out == "sigmoid":
            P = np.clip(P, 1e-12, 1 - 1e-12)
            return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        P = np.clip(P, 1e-12, 1.0)
        return -np.mean(np.sum(Y * np.log(P), axis=1))

    # ----- backward -----
    def gradients(self, X, Y):
        A, Z = self.forward(X)
        m = len(X)
        dZ = 2 * (A[-1] - Y) / m if self.out == "linear" else (A[-1] - Y) / m
        dg = ACT[self.act][1]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = A[l].T @ dZ
            gb[l] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dZ = (dZ @ self.W[l].T) * dg(Z[l - 1])
        return gW, gb

    # ----- training -----
    def fit(self, X, Y, epochs=200, lr=0.1, batch=None, opt="sgd", seed=0, beta=(0.9, 0.999)):
        rng = np.random.default_rng(seed)
        m, hist = len(X), []
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t = 0
        for _ in range(epochs):
            idx = rng.permutation(m) if batch else np.arange(m)
            for s in range(0, m, batch or m):
                j = idx[s:s + (batch or m)]
                gW, gb = self.gradients(X[j], Y[j])
                t += 1
                for l in range(len(self.W)):
                    if opt == "adam":
                        mW[l] = beta[0] * mW[l] + (1 - beta[0]) * gW[l]; vW[l] = beta[1] * vW[l] + (1 - beta[1]) * gW[l] ** 2
                        mb[l] = beta[0] * mb[l] + (1 - beta[0]) * gb[l]; vb[l] = beta[1] * vb[l] + (1 - beta[1]) * gb[l] ** 2
                        c1, c2 = 1 - beta[0] ** t, 1 - beta[1] ** t
                        self.W[l] -= lr * (mW[l] / c1) / (np.sqrt(vW[l] / c2) + 1e-8)
                        self.b[l] -= lr * (mb[l] / c1) / (np.sqrt(vb[l] / c2) + 1e-8)
                    else:
                        self.W[l] -= lr * gW[l]
                        self.b[l] -= lr * gb[l]
            hist.append(self.loss(X, Y))
        return hist

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))


def gradient_check(net, X, Y, eps=1e-6):
    """Largest relative error between backprop and central finite differences over all parameters."""
    gW, gb = net.gradients(X, Y)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for P, G in zip(params, grads):
            it = np.nditer(P, flags=["multi_index"])
            for _ in it:
                k = it.multi_index
                old = P[k]
                P[k] = old + eps; up = net.loss(X, Y)
                P[k] = old - eps; dn = net.loss(X, Y)
                P[k] = old
                num = (up - dn) / (2 * eps)
                worst = max(worst, abs(num - G[k]) / max(1e-8, abs(num) + abs(G[k])))
    return worst


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    for act in ("sigmoid", "tanh", "relu", "leaky_relu", "elu", "selu", "swish"):
        net = MLP([3, 5, 4, 1], act=act, out="linear", seed=1)
        err = gradient_check(net, X, rng.normal(size=(12, 1)))
        print(f"{act:10s} max relative gradient error = {err:.1e}")
        assert err < 1e-5
    for out, Y in (("sigmoid", rng.integers(0, 2, (12, 1)).astype(float)), ("softmax", np.eye(3)[rng.integers(0, 3, 12)])):
        net = MLP([3, 6, Y.shape[1]], act="tanh", out=out, seed=2)
        err = gradient_check(net, X, Y)
        print(f"output={out:8s} max relative gradient error = {err:.1e}")
        assert err < 1e-5
    print("backpropagation agrees with finite differences")
