# ci: run
"""Object-detection building blocks in NumPy: boxes, IoU, non-maximum suppression, YOLO-style
anchor targets, R-CNN box offsets, RPN anchors and average precision.

Boxes are rows (x1, y1, x2, y2) unless a function says "centre format" (cx, cy, w, h).
Image coordinates: x to the right, y downwards, as on slide 11 of the lecture.
"""
import numpy as np


# ---- boxes and IoU ---------
def xyxy_to_cxcywh(b):
    b = np.asarray(b, float)
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                     b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], axis=-1)


def cxcywh_to_xyxy(b):
    b = np.asarray(b, float)
    return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                     b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], axis=-1)


def iou_matrix(a, b):
    """IoU of every box in a (N,4) with every box in b (M,4): shape (N, M)."""
    a, b = np.atleast_2d(a).astype(float), np.atleast_2d(b).astype(float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    union = area(a)[:, None] + area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def iou(a, b):
    return float(iou_matrix(a, b)[0, 0])


# ---- non-maximum suppression ---------
def nms(boxes, scores, iou_thr=0.5, score_thr=0.0):
    """Greedy NMS. Repeatedly keep the box with the highest score and discard every remaining
    box whose IoU with it is GREATER than iou_thr. Returns the indices of the kept boxes."""
    order = [i for i in np.argsort(-scores) if scores[i] >= score_thr]
    keep = []
    while order:
        i = int(order.pop(0))
        keep.append(i)
        if order:
            ious = iou_matrix(boxes[i], boxes[order])[0]
            order = [j for j, v in zip(order, ious) if v <= iou_thr]
    return keep


def nms_per_class(boxes, scores, labels, iou_thr=0.5, score_thr=0.0):
    """NMS is run separately for every class, so a car cannot suppress a pedestrian."""
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], iou_thr, score_thr)]
    return sorted(keep, key=lambda i: -scores[i])


# ---- R-CNN style offsets (slides 57-58) ---------
def encode_offsets(prop, gt):
    """Offsets (dx, dy, dw, dh) moving a proposal onto a ground-truth box (centre format)."""
    px, py, pw, ph = prop
    gx, gy, gw, gh = gt
    return np.array([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)])


def decode_offsets(prop, d):
    px, py, pw, ph = prop
    return np.array([d[0] * pw + px, d[1] * ph + py, pw * np.exp(d[2]), ph * np.exp(d[3])])


# ---- YOLO-style targets: grid S x S, B anchors, C classes ---------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def anchor_shape_iou(wh, anchors):
    """IoU of a box with each anchor when both share one centre (only the shape counts)."""
    inter = np.minimum(wh[0], anchors[:, 0]) * np.minimum(wh[1], anchors[:, 1])
    return inter / (wh[0] * wh[1] + anchors[:, 0] * anchors[:, 1] - inter)


def encode_targets(gt_boxes, gt_labels, S, anchors, n_classes):
    """Target tensor y of shape (S, S, B, 5 + C) with entries [Pc, tx, ty, tw, th, one-hot class].
    gt_boxes: (N, 4) in centre format, normalised to [0, 1]; anchors: (B, 2) widths and heights in
    grid-cell units. Every object goes to the cell that contains its centre and to the anchor
    with the highest shape IoU (slide 46). tx, ty are logits of the offset inside the cell."""
    B = len(anchors)
    y = np.zeros((S, S, B, 5 + n_classes))
    for (cx, cy, w, h), c in zip(gt_boxes, gt_labels):
        gx, gy = cx * S, cy * S                       # centre in grid-cell units
        j, i = min(int(gx), S - 1), min(int(gy), S - 1)
        a = int(np.argmax(anchor_shape_iou(np.array([w * S, h * S]), anchors)))
        if y[i, j, a, 0] == 1:                 # slot already taken: the second object is lost
            continue
        off = np.clip([gx - j, gy - i], 1e-3, 1 - 1e-3)
        tw, th = np.log(w * S / anchors[a, 0]), np.log(h * S / anchors[a, 1])
        y[i, j, a, :5] = [1, *logit(off), tw, th]
        y[i, j, a, 5 + c] = 1
    return y


def decode_predictions(y, anchors, S, pc_thr=0.5):
    """Invert encode_targets: boxes (cx, cy, w, h) in [0, 1], objectness and class index for every
    slot with Pc > 0.5. Uses b = sigmoid(t) + c and b_wh = p * exp(t) (slide 37)."""
    out = []
    for i, j, a in zip(*np.where(y[..., 0] > pc_thr)):
        tx, ty, tw, th = y[i, j, a, 1:5]
        out.append((float((sigmoid(tx) + j) / S), float((sigmoid(ty) + i) / S),
                    float(anchors[a, 0] * np.exp(tw) / S), float(anchors[a, 1] * np.exp(th) / S),
                    float(y[i, j, a, 0]), int(np.argmax(y[i, j, a, 5:]))))
    return out


# ---- anchors of the region proposal network (Faster R-CNN) ---------
def rpn_anchors(fh, fw, stride=16, scales=(128, 256, 512), ratios=(0.5, 1.0, 2.0)):
    """k = len(scales) * len(ratios) anchors at every position of an fh x fw feature map, as (x1,y1,x2,y2).
    ratio = h / w; the area of an anchor is scale^2."""
    base = []
    for s in scales:
        for r in ratios:
            w, h = s / np.sqrt(r), s * np.sqrt(r)
            base.append([-w / 2, -h / 2, w / 2, h / 2])
    base = np.array(base)
    cx = (np.arange(fw) + 0.5) * stride
    cy = (np.arange(fh) + 0.5) * stride
    centres = np.stack(np.meshgrid(cx, cy), axis=-1).reshape(-1, 1, 2)
    return (centres.repeat(2, axis=2).reshape(-1, 1, 4) + base[None]).reshape(-1, 4)


# ---- average precision ---------
def average_precision(dets, gts, iou_thr=0.5):
    """All-point average precision of one class.
    dets: list of (image_id, score, box); gts: dict image_id -> (n, 4) array of ground-truth boxes.
    A detection is a true positive if its IoU with a not-yet-matched ground truth is >= iou_thr.
    Returns (AP, precision array, recall array)."""
    n_gt = sum(len(v) for v in gts.values())
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = []
    for img, score, box in sorted(dets, key=lambda d: -d[1]):
        g = gts.get(img, np.zeros((0, 4)))
        ok = 0
        if len(g):
            ious = iou_matrix(np.asarray(box)[None], g)[0]
            ious[used[img]] = -1
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                used[img][j] = True
                ok = 1
        tp.append(ok)
    tp = np.array(tp, float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):            # precision envelope
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), prec, rec


def mean_average_precision(dets_by_class, gts_by_class, iou_thr=0.5):
    aps = [average_precision(dets_by_class.get(c, []), gts_by_class[c], iou_thr)[0] for c in gts_by_class]
    return float(np.mean(aps)), aps


# ---- self-test ---------
if __name__ == "__main__":
    a, b = np.array([0, 0, 2, 2.]), np.array([1, 1, 3, 3.])
    assert abs(iou(a, b) - 1 / 7) < 1e-12 and iou(a, a) == 1
    bx = np.random.default_rng(0).uniform(0, 1, (5, 4)); bx[:, 2:] += bx[:, :2]
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(bx)), bx)
    prop, gt = np.array([0.5, 0.5, 0.2, 0.4]), np.array([0.55, 0.45, 0.3, 0.3])
    assert np.allclose(decode_offsets(prop, encode_offsets(prop, gt)), gt)
    anchors = np.array([[1.0, 2.0], [2.0, 1.0]])
    y = encode_targets(np.array([[0.7, 0.8, 0.5, 0.2]]), [1], 3, anchors, 3)
    (cx, cy, w, h, pc, c), = decode_predictions(y, anchors, 3)
    assert np.allclose([cx, cy, w, h], [0.7, 0.8, 0.5, 0.2]) and c == 1
    boxes = np.array([[0, 0, 2, 2], [0.1, 0, 2.1, 2], [5, 5, 7, 7.]])
    assert nms(boxes, np.array([0.8, 0.9, 0.7])) == [1, 2]
    gts = {0: np.array([[0, 0, 2, 2.]])}
    ap, _, _ = average_precision([(0, 0.9, [0, 0, 2, 2.])], gts)
    assert ap == 1.0
    print("detect_np self-test passed")
