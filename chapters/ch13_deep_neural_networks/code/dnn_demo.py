# ci: run
"""Numbers quoted in Chapter 13: hand example, linear collapse, XOR, vanishing gradients, initialisation."""
import warnings
import numpy as np
from sklearn.datasets import make_regression, make_moons
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from nn_numpy import MLP, ACT, softmax, sigmoid

warnings.filterwarnings("ignore")

# ---- 1. hand example: 2-2-1, ReLU hidden, linear output, MSE -----------------------------
print("== hand example (x=(1,2), y=1, alpha=0.1)")
w11, w13, b11 = 0.2, 0.4, 0.1
w12, w14, b12 = -0.5, 0.3, 0.2
w21, w22, b2 = 0.6, -0.1, 0.05
x1, x2, y = 1.0, 2.0, 1.0


def fwd(p):
    z1 = p["w11"] * x1 + p["w13"] * x2 + p["b11"]
    z2 = p["w12"] * x1 + p["w14"] * x2 + p["b12"]
    a1, a2 = max(0, z1), max(0, z2)
    return z1, z2, a1, a2, p["w21"] * a1 + p["w22"] * a2 + p["b2"]


p = dict(w11=w11, w13=w13, b11=b11, w12=w12, w14=w14, b12=b12, w21=w21, w22=w22, b2=b2)
z1, z2, a1, a2, yh = fwd(p)
L = (yh - y) ** 2
print(f"z1={z1:.2f} z2={z2:.2f} a1={a1:.2f} a2={a2:.2f} yhat={yh:.2f} loss={L:.4f}")
dy = 2 * (yh - y)
g = dict(w21=dy * a1, w22=dy * a2, b2=dy)
da1, da2 = dy * w21, dy * w22
dz1, dz2 = da1 * (z1 > 0), da2 * (z2 > 0)
g.update(w11=dz1 * x1, w13=dz1 * x2, b11=dz1, w12=dz2 * x1, w14=dz2 * x2, b12=dz2)
print("dL/dyhat =", round(dy, 4), " dL/da1, dL/da2 =", round(da1, 4), round(da2, 4))
print({k: round(v, 4) for k, v in g.items()})
p2 = {k: p[k] - 0.1 * g[k] for k in p}
print({k: round(v, 4) for k, v in p2.items()})
_, _, _, _, yh2 = fwd(p2)
print(f"after one step: yhat={yh2:.4f} loss={(yh2 - y) ** 2:.4f}")
net = MLP([2, 2, 1], act="relu", out="linear")
net.W = [np.array([[w11, w12], [w13, w14]]), np.array([[w21], [w22]])]
net.b = [np.array([[b11, b12]]), np.array([[b2]])]
gW, gb = net.gradients(np.array([[x1, x2]]), np.array([[y]]))
assert np.allclose(gW[1].ravel(), [g["w21"], g["w22"]]) and np.allclose(gW[0], [[g["w11"], g["w12"]], [g["w13"], g["w14"]]])
print("library network gives the same gradients")

# ---- 2. a network without activation is linear regression --------------------------------
print("== linear collapse on the notebook data (100 points, 2 features)")
X, yv = make_regression(100, n_features=2, noise=10, random_state=42)
X = (X - X.mean(0)) / X.std(0)
yv = ((yv - yv.mean()) / yv.std()).reshape(-1, 1)
ols = LinearRegression().fit(X, yv)
mse_ols = np.mean((ols.predict(X) - yv) ** 2)
lin = MLP([2, 2, 1], act="identity", out="linear", init=0.5, seed=0)
h = lin.fit(X, yv, epochs=2000, lr=0.05)
eff = lin.W[0] @ lin.W[1]
print(f"OLS mse={mse_ols:.5f} coef={ols.coef_.ravel().round(4)}; 2-2-1 linear net mse={h[-1]:.5f} W1@W2={eff.ravel().round(4)}")
rel = MLP([2, 2, 1], act="relu", out="linear", seed=3)
h = rel.fit(X, yv, epochs=2000, lr=0.05)
print(f"2-2-1 ReLU net mse={h[-1]:.5f} (data are linear, so no gain)")

# ---- 3. XOR -----------------------------------------------------------------------------------
print("== XOR")
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Yx = np.array([[0], [1], [1], [0]], float)
hand = MLP([2, 2, 1], act="relu", out="linear")
hand.W = [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[1.0], [-2.0]])]
hand.b = [np.array([[0.0, -1.0]]), np.array([[0.0]])]
print("hand-set ReLU net outputs:", hand.predict(Xx).ravel(), " loss", hand.loss(Xx, Yx))
best = min(MLP([2, 2, 1], act="identity", seed=s).fit(Xx, Yx, 3000, 0.1)[-1] for s in range(5))
print(f"best linear 2-2-1 loss after training: {best:.4f} (constant 0.5 gives 0.25)")
tn = MLP([2, 4, 1], act="tanh", out="sigmoid", init="xavier", seed=1)
tn.fit(Xx, Yx, 4000, 0.5)
print("tanh 2-4-1 + sigmoid, BCE:", tn.predict(Xx).ravel().round(3))

# ---- 4. vanishing gradients ---------------------------------------------------------------
print("== mean |dL/dW| per layer at initialisation, 10 hidden layers x 50 units, 256 samples")
rng = np.random.default_rng(0)
Xd = rng.normal(size=(256, 20))
Yd = rng.integers(0, 2, (256, 1)).astype(float)
for act, init in (("sigmoid", "xavier"), ("tanh", "xavier"), ("relu", "he")):
    n = MLP([20] + [50] * 10 + [1], act=act, out="sigmoid", init=init, seed=1)
    gW, _ = n.gradients(Xd, Yd)
    gm = [np.abs(w).mean() for w in gW]
    print(f" {act:8s} {init:7s} first layer {gm[0]:.2e}  last hidden {gm[-2]:.2e}  ratio {gm[-2] / gm[0]:.1f}")
print("max of sigmoid'(z) =", ACT["sigmoid"][1](np.array([0.0]))[0], " tanh'(0) =", ACT["tanh"][1](np.array([0.0]))[0])

# ---- 5. initialisation ---------------------------------------------------------------------------
print("== std of activations at the output of the 10th hidden layer (ReLU, inputs ~ N(0,1))")
Xi = rng.normal(size=(1000, 100))
for name, init in (("N(0,0.01^2)", 0.01), ("Xavier", "xavier"), ("He", "he")):
    n = MLP([100] + [100] * 10, act="relu", out="linear", init=init, seed=0)
    A, _ = n.forward(Xi)
    print(f" {name:12s} std of layer-9 activation {A[-2].std():.2e}  last layer pre-activation std {A[-1].std():.2e}")

# ---- 6. activation functions on two moons --------------------------------------------------------
print("== two moons (n=1000, noise 0.2), 2-16-16-1, 5 seeds, test accuracy")
Xm, ym = make_moons(1000, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym.reshape(-1, 1), test_size=0.3, random_state=0)
for act in ("sigmoid", "tanh", "relu", "leaky_relu", "elu", "selu", "swish"):
    accs = []
    for s in range(5):
        n = MLP([2, 16, 16, 1], act=act, out="sigmoid", init="he" if act in ("relu", "leaky_relu", "elu", "swish") else "xavier", seed=s)
        n.fit(Xtr, ytr, epochs=150, lr=0.05, batch=32, seed=s)
        accs.append(np.mean((n.predict(Xte) >= 0.5) == yte))
    print(f" {act:10s} mean {np.mean(accs):.3f}  min {np.min(accs):.3f}")
lin_n = MLP([2, 16, 16, 1], act="identity", out="sigmoid", seed=0)
lin_n.fit(Xtr, ytr, epochs=150, lr=0.05, batch=32)
print(f" no activation (identity) accuracy {np.mean((lin_n.predict(Xte) >= 0.5) == yte):.3f}")

# ---- 7. softmax and logits ------------------------------------------------------------------------
print("== softmax")
z = np.array([2.0, 4.0, 1.0, 0.5])
print("notebook logits", z, "->", softmax(z).round(4), "sum", softmax(z).sum())
big = np.array([1000.0, 1001.0, 1002.0])
naive = np.exp(big) / np.exp(big).sum()
print("naive softmax of (1000,1001,1002):", naive, " stable:", softmax(big).round(4))
print("sigmoid(-800) safe:", sigmoid(np.array([-800.0]))[0], " tanh formula (e^z-e^-z)/(e^z+e^-z) at z=800:", (np.exp(800.0) - np.exp(-800.0)) / (np.exp(800.0) + np.exp(-800.0)))

# ---- 8. counting parameters -----------------------------------------------------------------------
print("== parameters")
for sizes in ([2, 2, 1], [2, 10, 5, 1], [100, 10, 20, 10, 1], [784, 128, 64, 10]):
    print(sizes, MLP(sizes).n_params())

# ---- 9. the 2-2-1 classification network of the notebook ----------------------------------------
print("== notebook classification, 2-2-1 ReLU + sigmoid, 500 epochs, 400 training points")
from sklearn.datasets import make_classification

Xc, yc = make_classification(500, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
yc = yc.reshape(-1, 1)
Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=0.2, random_state=42)
for label, lr in (("lr 0.01 on the mean gradient", 0.01), ("lr 0.01 on the summed gradient (= 4.0 on the mean)", 4.0)):
    accs = []
    for s in range(10):
        n = MLP([2, 2, 1], act="relu", out="sigmoid", init=0.1, seed=s)
        n.fit(Xa, ya, epochs=500, lr=lr)
        accs.append(np.mean((n.predict(Xb) >= 0.5) == yb))
    print(f" {label}: mean accuracy {np.mean(accs):.3f}, min {np.min(accs):.3f}, max {np.max(accs):.3f}")

# ---- 10. width and approximation: one hidden ReLU layer fitting sin(x) ---------------------------
print("== 1-h-1 ReLU network fitting sin(x) on [-pi, pi] (200 points, Adam lr 0.02, 3000 full-batch steps)")
xs = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
ys = np.sin(xs)
for h in (2, 4, 8, 32):
    best = min(MLP([1, h, 1], act="relu", seed=s).fit(xs, ys, 3000, 0.02, opt="adam")[-1] for s in range(3))
    print(f" hidden units {h:3d}: parameters {3 * h + 1:4d}  best MSE of 3 seeds {best:.5f}")

# ---- 11. why cross-entropy goes with the sigmoid ---------------------------------------------------
print("== gradient with respect to the logit z for target y=1, prediction p=sigmoid(z)")
for z in (-8.0, -4.0, 0.0, 4.0):
    pz = sigmoid(np.array([z]))[0]
    print(f" z={z:5.1f} p={pz:.5f}  BCE grad = p-1 = {pz - 1:.5f}   MSE grad = 2(p-1)p(1-p) = {2 * (pz - 1) * pz * (1 - pz):.5f}")
