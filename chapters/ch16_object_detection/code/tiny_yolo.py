# ci: skip
"""A tiny YOLO-style detector trained from scratch on synthetic images (Chapter 16).

Images are 32x32 grey canvases with one or two shapes (class 0 square, 1 disc, 2 cross).
The network predicts, for every cell of a 4x4 grid and for B=2 anchors, the vector
[Pc, tx, ty, tw, th, c1, c2, c3] of slide 46; targets come from detect_np.encode_targets.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detect_np import (encode_targets, decode_predictions, nms_per_class, cxcywh_to_xyxy,
                       mean_average_precision, iou_matrix)

S, NC, IMG = 4, 3, 32
ANCHORS = np.array([[1.0, 1.0], [2.0, 2.0]])          # (w, h) in grid-cell units


def draw(rng):
    """Return an image (1, 32, 32) and a list of boxes (cx, cy, w, h) in [0, 1] with labels."""
    img = rng.normal(0, 0.05, (IMG, IMG))
    boxes, labels, cells = [], [], set()
    for _ in range(rng.integers(1, 3)):
        c = int(rng.integers(NC))
        size = int(rng.choice([6, 12]))
        cx, cy = rng.integers(size // 2 + 1, IMG - size // 2 - 1, 2)
        cell = (int(cy * S // IMG), int(cx * S // IMG))
        if cell in cells:
            continue
        cells.add(cell)
        x0, y0 = cx - size // 2, cy - size // 2
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        if c == 0:
            m = (xx >= x0) & (xx < x0 + size) & (yy >= y0) & (yy < y0 + size)
        elif c == 1:
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (size / 2) ** 2
        else:
            t = max(size // 4, 1)
            m = (((abs(xx - cx) <= t / 2) & (abs(yy - cy) <= size / 2)) |
                 ((abs(yy - cy) <= t / 2) & (abs(xx - cx) <= size / 2)))
        img[m] += 1.0
        boxes.append([cx / IMG, cy / IMG, size / IMG, size / IMG])
        labels.append(c)
    return img[None].astype(np.float32), np.array(boxes), labels


def make_set(n, seed):
    rng = np.random.default_rng(seed)
    X, Y, G = [], [], []
    for _ in range(n):
        img, b, l = draw(rng)
        X.append(img)
        Y.append(encode_targets(b, l, S, ANCHORS, NC).astype(np.float32))
        G.append((b, l))
    return np.stack(X), np.stack(Y), G


class TinyYolo(nn.Module):
    def __init__(self, B=2):
        super().__init__()
        self.B = B
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 4x4
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, B * (5 + NC), 1))
    def forward(self, x):
        o = self.body(x)                                  # (N, B*8, S, S)
        return o.permute(0, 2, 3, 1).reshape(x.shape[0], S, S, self.B, 5 + NC)


def yolo_loss(out, y):
    obj = y[..., 0]
    l_obj = F.binary_cross_entropy_with_logits(out[..., 0], obj, reduction="none")
    l_obj = (l_obj * (obj + 0.5 * (1 - obj))).sum()
    m = obj.unsqueeze(-1)
    l_xy = (m * (torch.sigmoid(out[..., 1:3]) - torch.sigmoid(y[..., 1:3])) ** 2).sum()
    l_wh = (m * (out[..., 3:5] - y[..., 3:5]) ** 2).sum()
    ce = F.cross_entropy(out[..., 5:].reshape(-1, NC), y[..., 5:].reshape(-1, NC).argmax(1),
                         reduction="none")
    l_cls = (obj * ce.reshape(obj.shape)).sum()
    return (l_obj + 5 * l_xy + 2 * l_wh + l_cls) / out.shape[0]


def detect(model, X, score_thr=0.05, iou_thr=0.5):
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X)).numpy()
    prob = np.exp(out[..., 5:] - out[..., 5:].max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)                    # softmax over the three classes
    out[..., 0] = 1 / (1 + np.exp(-out[..., 0])) * prob.max(-1)   # score = Pc * class confidence
    res = []
    for o in out:
        dets = decode_predictions(o, ANCHORS, S, pc_thr=score_thr)   # (cx, cy, w, h, score, class)
        if len(dets) == 0:
            res.append([]); continue
        d = np.array(dets)
        keep = d[:, 4] >= score_thr
        d = d[keep]
        if len(d) == 0:
            res.append([]); continue
        xy = cxcywh_to_xyxy(d[:, :4])
        k = nms_per_class(xy, d[:, 4], d[:, 5].astype(int), iou_thr)
        res.append([(xy[i], d[i, 4], int(d[i, 5])) for i in k])
    return res


def evaluate(model, X, G, iou_thr=0.5):
    res = detect(model, X)
    dets = {c: [] for c in range(NC)}
    gts = {c: {} for c in range(NC)}
    for n, ((b, l), r) in enumerate(zip(G, res)):
        for c in range(NC):
            gts[c][n] = np.array([cxcywh_to_xyxy(bb) for bb, ll in zip(b, l) if ll == c]).reshape(-1, 4)
        for box, sc, c in r:
            dets[c].append((n, sc, box))
    return mean_average_precision(dets, gts, iou_thr)


if __name__ == "__main__":
    torch.manual_seed(0)
    Xtr, Ytr, _ = make_set(1500, 1)
    Xte, Yte, Gte = make_set(300, 2)
    model = TinyYolo()
    print("parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    hist = []
    for ep in range(30):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 50):
            idx = perm[i:i + 50]
            loss = yolo_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / len(Xt))
        if ep % 5 == 4 or ep == 0:
            print(f"epoch {ep + 1:2d}  loss {hist[-1]:.3f}")
    m50, aps = evaluate(model, Xte, Gte, 0.5)
    m75, _ = evaluate(model, Xte, Gte, 0.75)
    print("test mAP@0.5 = %.3f  per class: %s" % (m50, np.round(aps, 3)))
    print("test mAP@0.75 = %.3f" % m75)
    np.save("tiny_yolo_loss.npy", np.array(hist))
    torch.save(model.state_dict(), "tiny_yolo.pt")
