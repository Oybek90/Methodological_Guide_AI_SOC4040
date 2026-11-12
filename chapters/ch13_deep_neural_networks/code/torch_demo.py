# ci: skip
"""PyTorch counterparts of the NumPy code (needs `pip install torch`): autograd check and the notebook experiments."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.set_num_threads(2)

# ---- 1. autograd reproduces the hand-computed gradients of the chapter example ---------------
lin1 = nn.Linear(2, 2)
lin2 = nn.Linear(2, 1)
with torch.no_grad():
    lin1.weight.copy_(torch.tensor([[0.2, 0.4], [-0.5, 0.3]]))   # rows: neuron a1, a2; columns: x1, x2
    lin1.bias.copy_(torch.tensor([0.1, 0.2]))
    lin2.weight.copy_(torch.tensor([[0.6, -0.1]]))
    lin2.bias.copy_(torch.tensor([0.05]))
x, y = torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0]])
loss = nn.functional.mse_loss(lin2(torch.relu(lin1(x))), y)
loss.backward()
print("loss", round(loss.item(), 4))
print("dL/dW1 (rows a1,a2):", np.round(lin1.weight.grad.numpy().astype(float), 3).tolist(), " dL/db1:", np.round(lin1.bias.grad.numpy().astype(float), 3))
print("dL/dW2:", np.round(lin2.weight.grad.numpy().astype(float), 3), " dL/db2:", np.round(lin2.bias.grad.numpy().astype(float), 3))


def train(model, Xtr, ytr, Xte, yte, lr, epochs, loss_fn, logits=False):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(Xt), yt).backward()
        opt.step()
    with torch.no_grad():
        out = model(Xe)
        p = torch.sigmoid(out) if logits else out
    return float(np.mean((p.numpy() >= 0.5) == yte))


# ---- 2. the notebook's two-feature classification problem -------------------------------------
X, y = make_classification(500, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
y = y.reshape(-1, 1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print("== 2 features, 500 samples")
for name, mk in (
    ("2-10-5-1 ReLU/LeakyReLU + Sigmoid + BCELoss (notebook)", lambda: nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 5), nn.LeakyReLU(), nn.Linear(5, 1), nn.Sigmoid())),
):
    accs = []
    for s in range(5):
        torch.manual_seed(s)
        accs.append(train(mk(), Xtr, ytr, Xte, yte, 0.1, 1000, nn.BCELoss()))
    print(f" {name}: mean test accuracy {np.mean(accs):.3f} (min {np.min(accs):.3f})")
accs = []
for s in range(5):
    torch.manual_seed(s)
    m = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 5), nn.LeakyReLU(), nn.Linear(5, 1))
    accs.append(train(m, Xtr, ytr, Xte, yte, 0.1, 1000, nn.BCEWithLogitsLoss(), logits=True))
print(f" same net, logits + BCEWithLogitsLoss: mean {np.mean(accs):.3f}")

# ---- 3. the notebook's 100-feature problem (2 informative features) ----------------------------
X, y = make_classification(10000, n_features=100, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
y = y.reshape(-1, 1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print("== 100 features, 10000 samples, 100-10-20-10-1 (ReLU, Tanh, SELU), SGD lr 0.01, 1000 full-batch epochs")
accs = []
for s in range(3):
    torch.manual_seed(s)
    m = nn.Sequential(nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 20), nn.Tanh(), nn.Linear(20, 10), nn.SELU(), nn.Linear(10, 1))
    accs.append(train(m, Xtr, ytr, Xte, yte, 0.01, 1000, nn.BCEWithLogitsLoss(), logits=True))
print(f" test accuracy over 3 seeds: {np.round(accs, 3)}; trivial majority baseline {max(yte.mean(), 1 - yte.mean()):.3f}")
print("parameters:", sum(p.numel() for p in m.parameters()))
