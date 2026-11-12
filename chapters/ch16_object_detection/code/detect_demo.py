# ci: run
"""Numbers quoted in Chapter 16 (IoU, NMS, anchors, offsets, sliding window, AP)."""
import numpy as np
from detect_np import *
from cnn_numpy import conv2d, pool2d

print("== 1. Slide 11: bx=0.5, by=0.7, bh=0.3, bw=0.4 (centre format, image coordinates in [0,1])")
box = np.array([0.5, 0.7, 0.4, 0.3])          # cx, cy, w, h
print("corners (x1,y1,x2,y2):", cxcywh_to_xyxy(box))
pred = cxcywh_to_xyxy(np.array([0.55, 0.72, 0.4, 0.3]))
print("IoU with a prediction shifted by (0.05, 0.02): %.3f" % iou(cxcywh_to_xyxy(box), pred))
a, b = np.array([0, 0, 2, 2.]), np.array([1, 1, 3, 3.])
print("two 2x2 squares overlapping in a 1x1 square: IoU =", iou(a, b), "= 1/7")

print("\n== 2. Non-max suppression with scores 0.9, 0.8, 0.7 (slides 21-23)")
boxes = np.array([[0.30, 0.30, 0.70, 0.70], [0.25, 0.28, 0.62, 0.66], [0.38, 0.33, 0.78, 0.74]])
scores = np.array([0.9, 0.8, 0.7])
print("pairwise IoU with the best box:", np.round(iou_matrix(boxes[0], boxes)[0], 2))
print("kept (IoU threshold 0.5):", nms(boxes, scores, 0.5))
print("wrong rule of slide 23 (discard IoU <= 0.5), kept:", [0] + [i for i in (1, 2) if iou_matrix(boxes[0], boxes[i][None])[0, 0] > 0.5])
two = np.vstack([boxes, [[0.05, 0.05, 0.25, 0.30]]])
print("a second, separate object survives:", nms(two, np.array([0.9, 0.8, 0.7, 0.6]), 0.5))

print("\n== 3. Bounding-box offsets (slides 57-58) and the swapped formula on slide 57")
prop, gt = np.array([0.50, 0.50, 0.20, 0.40]), np.array([0.55, 0.45, 0.30, 0.30])
d = encode_offsets(prop, gt)
print("offsets dx,dy,dw,dh:", np.round(d, 4), "-> decoded:", np.round(decode_offsets(prop, d), 4))
bad = np.array([d[0] * prop[2] + prop[0], d[1] * prop[3] + prop[1], prop[3] * np.exp(d[2]), prop[2] * np.exp(d[3])])
print("slide-57 formula (w and h of the proposal exchanged):", np.round(bad, 4))

print("\n== 4. Anchors (slides 33-35): IoU of the ground truth with three anchors, best anchor")
print("IoUs on slide 35: 0.75, 0.80, 0.45 -> anchor", int(np.argmax([0.75, 0.80, 0.45])) + 1)
anch = np.array([[1.0, 2.0], [2.0, 1.0]])
print("shape IoU of a 0.5x0.2 (image units) car on a 3x3 grid with anchors", anch.tolist(), ":",
      np.round(anchor_shape_iou(np.array([0.5 * 3, 0.2 * 3]), anch), 3))
y = encode_targets(np.array([[0.7, 0.8, 0.5, 0.2]]), [1], 3, anch, 3)
print("target tensor shape", y.shape, "=", y.size, "numbers; non-zero slot:", np.argwhere(y[..., 0] == 1).tolist())
print("slide 41: 3x3 grid, 2 anchors, 8 numbers each ->", 3 * 3 * 2 * 8, "outputs")
print("decoded:", np.round(decode_predictions(y, anch, 3)[0], 3))

print("\n== 5. Faster R-CNN anchors for a 1000 x 600 image, stride 16")
fh, fw = 38, 63
A = rpn_anchors(fh, fw)
inside = ((A[:, 0] >= 0) & (A[:, 1] >= 0) & (A[:, 2] <= 1000) & (A[:, 3] <= 600)).sum()
print(f"{fh}x{fw} positions x 9 = {len(A)} anchors, {inside} lie completely inside the image")
print("anchor sizes at one position (w x h):", [f"{a[2]-a[0]:.0f}x{a[3]-a[1]:.0f}" for a in A[:9]])

print("\n== 6. Sliding window: a ConvNet on 14x14 windows applied to a 16x16 image")
rng = np.random.default_rng(0)
w1, w2 = rng.normal(size=(4, 1, 3, 3)), rng.normal(size=(8, 4, 6, 6))   # conv, and 'dense layer' as 6x6 conv
net = lambda x: conv2d(pool2d(np.maximum(conv2d(x, w1), 0), 2, 2), w2)
img = rng.normal(size=(1, 1, 16, 16))
full = net(img)
crops = np.stack([[net(img[:, :, r:r + 14, c:c + 14])[0, :, 0, 0] for c in (0, 2)] for r in (0, 2)])
print("one pass over the 16x16 image gives output shape", full.shape)
print("same as 4 separate 14x14 windows (stride 2):", np.allclose(full[0].transpose(1, 2, 0), crops))
m1 = 4 * 9 * 12 * 12; m2 = 8 * 4 * 36
print("multiply-adds per 14x14 window:", m1 + m2, "; 4 windows:", 4 * (m1 + m2),
      "; one shared pass:", 4 * 9 * 14 * 14 + 8 * 4 * 36 * 4)
img2 = rng.normal(size=(1, 1, 28, 28))
nw = (28 - 14) // 2 + 1
print(f"28x28 image: {nw}x{nw} windows; separate: {nw * nw * (m1 + m2)} multiply-adds; one shared pass:",
      26 * 26 * 36 + nw * nw * 8 * 4 * 36, "(%.1fx fewer)" % (nw * nw * (m1 + m2) / (26 * 26 * 36 + nw * nw * 8 * 4 * 36)))
H = 448
print(f"window 224 stride 8 on a {H}x{H} image: {((H - 224) // 8 + 1) ** 2} windows")

print("\n== 7. Average precision")
gts = {0: np.array([[0, 0, 2, 2.], [5, 5, 7, 7.]]), 1: np.array([[1, 1, 3, 3.]])}
dets = [(0, 0.95, [0, 0, 2, 2.]), (1, 0.9, [1.1, 1, 3.1, 3]), (0, 0.8, [5, 5, 6, 6]),
        (1, 0.6, [4, 4, 6, 6.]), (0, 0.5, [5.2, 5.1, 7.1, 7.0])]
ap, p, r = average_precision(dets, gts)
print("precision:", np.round(p, 3), "recall:", np.round(r, 3), "AP50 = %.4f" % ap)
print("IoU of detection 3 (score 0.8) with its ground truth: %.2f" % iou([5, 5, 6, 6.], [5, 5, 7, 7.]))
print("AP at IoU threshold 0.9: %.4f" % average_precision(dets, gts, 0.9)[0])
