# ci: skip
"""Figures for Chapter 16 (grayscale-safe vector PDFs)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from detect_np import iou, nms, average_precision
import tiny_yolo as ty

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def box(ax, b, ls="-", lw=1.6, color="k", label=None, fill=None):
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=fill is not None, fc=fill,
                           alpha=0.35 if fill else 1, ec=color, ls=ls, lw=lw, label=label))


# 1. IoU for three overlaps ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
gt = [0, 0, 4, 4.]
for ax, pr in zip(axs, ([0.3, 0.2, 4.2, 4.1], [1.5, 1, 5.5, 5], [3.2, 3.2, 7, 7])):
    box(ax, gt, "-", 1.8, "k", fill="0.6")
    box(ax, pr, "--", 1.8, "k")
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("IoU = %.2f" % iou(gt, pr))
save(fig, "fig_iou.pdf")

# 2. NMS before / after ---------------------------------------------------------------------------------
bx = np.array([[1, 1, 5, 5], [1.4, 1.2, 5.4, 5.3], [0.8, 0.7, 4.7, 4.8], [7, 4, 10, 7], [7.3, 4.3, 10.4, 7.2]])
sc = np.array([0.7, 0.9, 0.6, 0.8, 0.55])
keep = nms(bx, sc, 0.5)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, ttl, idx in zip(axs, ("before NMS", "after NMS (threshold 0.5)"), (range(5), keep)):
    for i in idx:
        box(ax, bx[i], "-" if i in keep else ":", 1.8)
        ax.text(bx[i][0], bx[i][1] - 0.15, "%.2f" % sc[i], fontsize=8)
    ax.set_xlim(0, 11); ax.set_ylim(8, 0); ax.set_aspect("equal"); ax.set_title(ttl); ax.axis("off")
save(fig, "fig_nms.pdf")

# 3. YOLO grid and anchors ---------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 3.6))
for k in range(4):
    ax.plot([k, k], [0, 3], color="0.7", lw=0.8); ax.plot([0, 3], [k, k], color="0.7", lw=0.8)
obj_c = (1.65, 1.35); w, h = 1.3, 0.8
box(ax, [obj_c[0] - w / 2, obj_c[1] - h / 2, obj_c[0] + w / 2, obj_c[1] + h / 2], "-", 2)
ax.add_patch(Rectangle((1, 1), 1, 1, fill=True, fc="0.85", ec="k", lw=1.2))
for (aw, ah), ls in zip(((1.0, 1.0), (1.6, 0.8)), (":", "--")):
    box(ax, [1.5 - aw / 2, 1.5 - ah / 2, 1.5 + aw / 2, 1.5 + ah / 2], ls, 1.3)
ax.plot(*obj_c, "ko"); ax.plot(1.5, 1.5, "k+", ms=9)
ax.annotate(r"$\sigma(t_x),\sigma(t_y)$", xy=obj_c, xytext=(2.0, 0.35), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax.set_xlim(-0.1, 3.1); ax.set_ylim(3.1, -0.1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
save(fig, "fig_yolo_grid.pdf")

# 4. precision-recall curve for the AP example -----------------------------------------------------------
gts = {0: np.array([[0, 0, 2, 2.], [5, 5, 7, 7.]]), 1: np.array([[1, 1, 3, 3.]])}
dets = [(0, 0.95, [0, 0, 2, 2.]), (1, 0.9, [1.1, 1, 3.1, 3]), (0, 0.8, [5, 5, 6, 6]),
        (1, 0.6, [4, 4, 6, 6.]), (0, 0.5, [5.2, 5.1, 7.1, 7.0])]
ap, p, r = average_precision(dets, gts)
env = np.maximum.accumulate(p[::-1])[::-1]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.plot(r, p, "ko", label="measured")
ax.step(np.r_[0, r], np.r_[env[0], env], where="pre", color="k", label="interpolated")
ax.fill_between(np.r_[0, r], np.r_[env[0], env], step="pre", color="0.85")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
ax.set_title("AP$_{50}$ = %.3f" % ap); ax.legend(frameon=False, loc="lower left")
save(fig, "fig_pr.pdf")

# 5. tiny YOLO: training loss and detections on test images -----------------------------------------------
torch.manual_seed(0)
model = ty.TinyYolo(); model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "tiny_yolo.pt")))
hist = np.load(os.path.join(os.path.dirname(__file__), "tiny_yolo_loss.npy"))
X, Y, G = ty.make_set(8, 7)
res = ty.detect(model, X, score_thr=0.4)
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8))
for ax, x, (gb, gl), r in zip(axs.ravel(), X, G, res):
    ax.imshow(x[0], cmap="gray_r"); ax.axis("off")
    for b, l in zip(gb, gl):
        c = ty.IMG
        box(ax, [(b[0] - b[2] / 2) * c, (b[1] - b[3] / 2) * c, (b[0] + b[2] / 2) * c, (b[1] + b[3] / 2) * c],
            ":", 1.4, "0.4")
    for bb, s, cl in r:
        box(ax, np.array(bb) * ty.IMG, "-", 1.4)
        ax.text(bb[0] * ty.IMG, bb[1] * ty.IMG - 0.5, "%s %.2f" % ("SDX"[cl], s), fontsize=6)
save(fig, "fig_tiny_yolo.pdf")
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.plot(np.arange(1, len(hist) + 1), hist, "k-"); ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
save(fig, "fig_tiny_loss.pdf")

# 6. sliding-window output map ----------------------------------------------------------------------------------
rng = np.random.default_rng(3)
M = rng.uniform(0, 0.2, (8, 8)); M[2:4, 3:5] += [[0.5, 0.7], [0.6, 0.8]]
fig, ax = plt.subplots(figsize=(3.2, 3.2))
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.set_title("window scores, $8\\times8$ map")
fig.colorbar(im, ax=ax, fraction=0.046)
save(fig, "fig_sliding.pdf")
print("figures written to", os.path.abspath(OUT))
