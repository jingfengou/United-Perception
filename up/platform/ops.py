"""Shared detection helpers; complete native operators live behind NativeOps."""
import torch
from up.runtime import get_runtime
from .torch_native import overlap as box_overlap, sigmoid_focal as sigmoid_focal_loss


def nms(boxes, scores, threshold):
    """YOLO11/YOLOX 实际使用的连续坐标 NMS；公共调用入口由我们维护。

    boxes[N,4] 为 xyxy、scores[N] 为分数、threshold 为 IoU 阈值；
    返回输入设备上的 int64 原框索引[K]，无反向要求。这里没有 +1 offset。
    NVIDIA 直接用 torchvision.ops.nms，其他厂商已使用下面的共享 Torch 实现。
    负责人先验证本卡的排序、边界/空输入和性能；需要专用 SDK 时交付同签名
    实现和测试，我们接入此公共入口。修改 NativeOps.naive_nms 不会改变此路线。
    """
    # NVIDIA 沿用 TorchVision；其他厂商在输入设备上使用普通 Tensor 运算完成筛选。
    if get_runtime().platform == 'nvidia':
        from torchvision.ops import nms as torchvision_nms
        return torchvision_nms(boxes, scores, threshold)

    # Same-device reference path. No CUDA binary or implicit CPU fallback.
    order = scores.argsort(descending=True)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.numel():
        index = order[0]
        keep.append(index)
        remaining = order[1:]
        lt = torch.maximum(boxes[index, :2], boxes[remaining, :2])
        rb = torch.minimum(boxes[index, 2:], boxes[remaining, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        iou = intersection / (area[index] + area[remaining] - intersection)
        order = remaining[~(iou > threshold)]
    return torch.stack(keep) if keep else order
