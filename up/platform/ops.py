import torch
import torch.nn.functional as F

from up.runtime import get_runtime


def box_overlap(b1, b2, mode='IoU'):
    # 保持 UP 原扩展的闭区间坐标语义；与下方 NMS 使用的连续坐标分开处理。
    # Match csrc/iou_overlap exactly: inclusive coordinates and denominator >= 1.
    area1 = ((b1[:, 2:4] - b1[:, :2]) + 1).prod(dim=1)
    area2 = ((b2[:, 2:4] - b2[:, :2]) + 1).prod(dim=1)
    lt = torch.maximum(b1[:, None, :2], b2[:, :2])
    rb = torch.minimum(b1[:, None, 2:4], b2[:, 2:4])
    intersection = (rb - lt + 1).clamp(min=0).prod(dim=2)
    denominator = area1[:, None] + area2 - intersection if mode == 'IoU' else area1[:, None]
    return intersection / denominator.clamp(min=1)


def nms(boxes, scores, threshold):
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


def sigmoid_focal_loss(input, target, normalizer, gamma, alpha, num_channels, reduction):
    # 为不能加载 NVIDIA 扩展的平台提供可自动求导的 Torch 实现，输出仍按 UP 约定。
    # UP labels: background=0, foreground=1..C, ignored=-1.
    classes = torch.arange(1, num_channels + 1, device=input.device)
    positive = (target[:, None] == classes).to(input.dtype)
    probability = input.sigmoid()
    ce = F.binary_cross_entropy_with_logits(input, positive, reduction='none')
    pt = probability * positive + (1 - probability) * (1 - positive)
    alpha_t = alpha * positive + (1 - alpha) * (1 - positive)
    loss = ce * (1 - pt).pow(gamma) * alpha_t
    loss = loss * (target[:, None] != -1).to(input.dtype) / normalizer.clamp(min=1)
    return loss if reduction == 'none' else loss.sum().reshape(1)
