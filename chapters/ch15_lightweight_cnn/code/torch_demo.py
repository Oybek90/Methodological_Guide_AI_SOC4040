# ci: skip
"""PyTorch experiments of Chapter 15 on sklearn's 8x8 digits (offline, CPU, a few seconds each).

1. standard vs depthwise separable network: parameters, multiply-adds, test accuracy (5 seeds);
2. batch normalization vs none, plain SGD, three learning rates (3 seeds);
3. the PyTorch layers agree with the NumPy ones, and the parameter counts of the lecture layers.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(2)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 1, 8, 8).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))


def conv_block(c_in, c_out, separable, bn):
    if separable:
        layers = [nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=not bn),   # depthwise
                  nn.Conv2d(c_in, c_out, 1, bias=not bn)]                            # pointwise
    else:
        layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return layers + [nn.ReLU()]


def make_net(separable=False, bn=True, width=(16, 32)):
    layers = conv_block(1, width[0], False, bn) + conv_block(width[0], width[1], separable, bn)
    return nn.Sequential(*layers, nn.MaxPool2d(2), nn.Flatten(), nn.Linear(width[1] * 16, 10))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, size=(1, 1, 8, 8)):
    """Multiply-adds of conv and linear layers for one image (biases and BN not counted)."""
    total = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total.append(mod.weight.numel() * out.shape[2] * out.shape[3])
        elif isinstance(mod, nn.Linear):
            total.append(mod.weight.numel())
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(size))
    for h in hs:
        h.remove()
    return sum(total)


def train(model, lr, epochs, opt="adam", bs=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.optim.Adam(model.parameters(), lr=lr) if opt == "adam" else torch.optim.SGD(model.parameters(), lr=lr)
    lossf, hist = nn.CrossEntropyLoss(), []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            o.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            o.step()
        model.eval()
        with torch.no_grad():
            hist.append(lossf(model(Xtr), ytr).item())
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc, hist


if __name__ == "__main__":
    print("== 1. standard vs separable (Adam, lr 0.01, 30 epochs, BN, 5 seeds)")
    for sep in (False, True):
        accs = []
        for s in range(5):
            torch.manual_seed(s)
            net = make_net(separable=sep)
            accs.append(train(net, 0.01, 30, seed=s)[0])
        net = make_net(separable=sep)
        print(f"{'separable' if sep else 'standard '}: params {n_params(net):5d}  MACs {count_macs(net):6d}  "
              f"test acc {np.mean(accs):.3f} +- {np.std(accs):.3f}")

    print("\n== 2. batch normalization, plain SGD, 15 epochs, 3 seeds (final training loss, test acc)")
    for lr in (0.01, 0.1, 0.5):
        row = []
        for bn in (False, True):
            res = []
            for s in range(3):
                torch.manual_seed(s)
                acc, h = train(make_net(bn=bn), lr, 15, opt="sgd", seed=s)
                res.append((h[-1] if np.isfinite(h[-1]) else np.nan, acc))
            r = np.array(res)
            row.append(f"{'BN   ' if bn else 'no BN'} loss {np.nanmean(r[:, 0]):.3f} acc {r[:, 1].mean():.3f}")
        print(f"lr {lr:<4}: " + " | ".join(row))

    print("\n== 3. parameter counts of the lecture layers (32 -> 64 channels, 3x3)")
    std = nn.Conv2d(32, 64, 3)
    grp = nn.Conv2d(32, 64, 3, groups=32)
    sep = nn.Sequential(nn.Conv2d(32, 32, 3, groups=32), nn.Conv2d(32, 64, 1))
    print("standard", n_params(std), "grouped(32)", n_params(grp), "separable", n_params(sep))
    bn = nn.BatchNorm2d(64)
    print("BatchNorm2d(64): trainable", n_params(bn), "buffers", sum(b.numel() for b in bn.buffers()),
          "(running_mean, running_var, num_batches_tracked)")

    # NumPy and PyTorch agree
    from lightcnn import grouped_conv2d, batchnorm_forward
    rng = np.random.default_rng(0)
    xn, wn = rng.normal(size=(2, 4, 7, 7)), rng.normal(size=(6, 2, 3, 3))
    conv = nn.Conv2d(4, 6, 3, groups=2, padding=1, bias=False)
    conv.weight.data = torch.tensor(wn)
    d1 = np.abs(conv(torch.tensor(xn)).detach().numpy() - grouped_conv2d(xn, wn, groups=2, pad=1)).max()
    b = nn.BatchNorm2d(4, eps=1e-5).double()
    d2 = np.abs(b(torch.tensor(xn)).detach().numpy() - batchnorm_forward(xn, np.ones(4), np.zeros(4))[0]).max()
    print(f"max difference grouped conv {d1:.1e}, batch norm {d2:.1e}")
