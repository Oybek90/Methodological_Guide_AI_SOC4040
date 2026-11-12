# ci: skip
"""PyTorch checks and the small experiment of Chapter 14 (needs torch; run by hand)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from cnn_numpy import conv2d

torch.set_num_threads(2)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# 1. the NumPy convolution equals nn.Conv2d ---------------------------------------------------
torch.manual_seed(0)
conv = nn.Conv2d(3, 4, 3, stride=2, padding=1)
x = torch.randn(2, 3, 9, 9)
ref = conv(x).detach().numpy()
mine = conv2d(x.numpy().astype(float), conv.weight.detach().numpy().astype(float),
              conv.bias.detach().numpy().astype(float), stride=2, pad=1)
print("max |NumPy - PyTorch| convolution:", float(np.abs(ref - mine).max()))

# 2. architectures of the notebook ------------------------------------------------------------


def sep(c_in, c_out, k=3, s=1, p=1, depthwise=True):
    return nn.Sequential(nn.Conv2d(c_in, c_in, k, s, p, groups=c_in if depthwise else 1),
                         nn.Conv2d(c_in, c_out, 1))


lenet_nb = nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.AvgPool2d(5, 2), nn.Conv2d(6, 16, 3), nn.ReLU(),
                         nn.AvgPool2d(2, 2), nn.Flatten(), nn.Linear(256, 120), nn.ReLU(), nn.Linear(120, 84),
                         nn.ReLU(), nn.Linear(84, 10))
lenet_cl = nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.AvgPool2d(2, 2), nn.Conv2d(6, 16, 5), nn.ReLU(),
                         nn.AvgPool2d(2, 2), nn.Flatten(), nn.Linear(256, 120), nn.ReLU(), nn.Linear(120, 84),
                         nn.ReLU(), nn.Linear(84, 10))
z = torch.zeros(1, 1, 28, 28)
print("LeNet notebook:", n_params(lenet_nb), tuple(lenet_nb(z).shape), "| classic:", n_params(lenet_cl),
      tuple(lenet_cl(z).shape))


def std_net(block):
    return nn.Sequential(block(1, 32), block(32, 64), nn.MaxPool2d(2), block(64, 128), block(128, 256),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(256 * 7 * 7, 50), nn.ReLU(), nn.Linear(50, 10))


conv3 = lambda a, b: nn.Conv2d(a, b, 3, padding=1)
print("myStdCNN:", n_params(std_net(conv3)), "| DWS as in the notebook (no groups):",
      n_params(std_net(lambda a, b: sep(a, b, depthwise=False))), "| DWS with groups=C:",
      n_params(std_net(lambda a, b: sep(a, b))))

alex = nn.Sequential(nn.Conv2d(1, 96, 11, 4, 2), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Conv2d(96, 256, 5, 1, 2),
                     nn.ReLU(), nn.MaxPool2d(3, 2), nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
                     nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(), nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),
                     nn.MaxPool2d(3, 2), nn.Flatten(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
                     nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 10))
print("AlexNet (5 conv):", n_params(alex), "output for 224x224:", tuple(alex(torch.zeros(1, 1, 224, 224)).shape))
try:
    alex(z)
except Exception as e:
    print("AlexNet on 28x28 fails:", str(e)[:90])

# 3. MLP vs CNN on the 8x8 digits, also on shifted test images ------------------------------------
data = load_digits()
Xall = (data.images / 16.0).astype("float32")[:, None]
yall = data.target


def shift(a, dx):
    return np.roll(a, dx, axis=3)


def make_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def make_cnn():
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                         nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 2 * 2, 10))


def run(make, seed):
    torch.manual_seed(seed)
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.25, random_state=seed, stratify=yall)
    Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))
    net = make()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(40):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            nn.functional.cross_entropy(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        acc = (net(Xte).argmax(1) == yte).float().mean().item()
        acc_s = (net(torch.tensor(shift(Xte.numpy(), 1))).argmax(1) == yte).float().mean().item()
    return acc, acc_s, n_params(net)


for name, make in [("MLP 64-64-10", make_mlp), ("CNN 8-16 conv", make_cnn)]:
    r = np.array([run(make, s) for s in range(5)])
    print(f"{name}: params {int(r[0, 2])}, test acc {r[:, 0].mean():.3f} (min {r[:, 0].min():.3f}),"
          f" 1-pixel-shifted test acc {r[:, 1].mean():.3f}")
