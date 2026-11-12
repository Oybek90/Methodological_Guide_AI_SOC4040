# ci: skip
"""Torchvision detection models (Chapter 16): parameter counts and output structure.
Weights are NOT downloaded (weights=None), so the predictions of the untrained models are meaningless;
only the shapes and the API are of interest."""
import torch
import torchvision
from torchvision.models import detection as det
from torchvision.ops import box_iou, nms

torch.manual_seed(0)
print("torchvision", torchvision.__version__)

# 1. IoU and NMS from torchvision.ops agree with our NumPy functions
a = torch.tensor([[0., 0, 2, 2]]); b = torch.tensor([[1., 1, 3, 3]])
print("box_iou:", box_iou(a, b).item(), "(1/7 = %.4f)" % (1 / 7))
boxes = torch.tensor([[0, 0, 2, 2], [0.1, 0, 2.1, 2], [5, 5, 7, 7.]])
print("nms keep:", nms(boxes, torch.tensor([0.8, 0.9, 0.7]), 0.5).tolist())

# 2. parameter counts of the architectures named on the slides
models = {
    "Faster R-CNN ResNet-50 FPN": det.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None),
    "RetinaNet ResNet-50 FPN": det.retinanet_resnet50_fpn(weights=None, weights_backbone=None),
    "SSD300 VGG-16": det.ssd300_vgg16(weights=None, weights_backbone=None),
    "SSDlite320 MobileNetV3": det.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None),
}
for name, m in models.items():
    print(f"{name:30s} {sum(p.numel() for p in m.parameters()) / 1e6:6.1f} M parameters")

# 3. the output of a detector in evaluation mode: a list of dicts with boxes, labels, scores
m = det.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=4).eval()
with torch.no_grad():
    out = m([torch.rand(3, 320, 320)])
print({k: tuple(v.shape) for k, v in out[0].items()})

# 4. training mode: targets are dicts with boxes (x1, y1, x2, y2 in pixels) and labels (1..C; 0 = background)
m.train()
img = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]   # BatchNorm needs a batch of at least 2
tgt = [{"boxes": torch.tensor([[50., 60, 150, 200]]), "labels": torch.tensor([2])}] * 2
losses = m(img, tgt)
print("training losses:", {k: round(v.item(), 3) for k, v in losses.items()})
