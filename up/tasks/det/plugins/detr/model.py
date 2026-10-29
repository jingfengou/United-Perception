import torch
from torch import nn
from torchvision.models import resnet50

from up.utils.general.global_flag import ALIGNED_FLAG
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .official.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
from .official.detr import DETR, PostProcess, SetCriterion
from .official.matcher import HungarianMatcher
from .official.misc import NestedTensor
from .official.position_encoding import PositionEmbeddingSine
from .official.transformer import Transformer


@MODULE_ZOO_REGISTRY.register('detr')
class DetrForUP(nn.Module):
    """Official DETR with UP's batch, label and bounding-box conventions.

    ``num_classes`` includes UP's background class; DETR adds its own no-object
    class after the foreground classes. Images must use top-left alignment.
    """

    def __init__(self, num_classes=81, num_queries=100, aux_loss=True):
        super().__init__()
        body = resnet50(pretrained=False, norm_layer=FrozenBatchNorm2d)
        backbone = BackboneBase(body, train_backbone=True, num_channels=2048,
                                return_interm_layers=False)
        backbone = Joiner(backbone, PositionEmbeddingSine(128, normalize=True))
        backbone.num_channels = 2048
        transformer = Transformer(d_model=256, nhead=8, num_encoder_layers=6,
                                  num_decoder_layers=6, dim_feedforward=2048,
                                  dropout=0.1, return_intermediate_dec=True)
        self.detector = DETR(backbone, transformer, num_classes - 1, num_queries, aux_loss)
        self.weights = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        if aux_loss:
            self.weights.update({name + '_' + str(i): weight
                                 for i in range(5) for name, weight in list(self.weights.items())})
        self.criterion = SetCriterion(num_classes - 1, HungarianMatcher(1, 5, 2),
                                      self.weights, 0.1, ['labels', 'boxes', 'cardinality'])
        self.postprocess = PostProcess()

    def forward(self, batch):
        images, information = batch['image'], batch['image_info']
        mask = images.new_ones((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool)
        for index, info in enumerate(information):
            if float(info[6]) or float(info[7]):
                raise ValueError('DETR requires top-left aligned images without left/top padding.')
            mask[index, :int(info[0]), :int(info[1])] = False
        predictions = self.detector(NestedTensor(images, mask))
        if self.training:
            targets = []
            for boxes, info in zip(batch['gt_bboxes'], information):
                boxes = boxes[boxes[:, 4] > 0]
                height, width = float(info[0]), float(info[1])
                scale = info[2]
                sh, sw = scale if isinstance(scale, (tuple, list)) else (scale, scale)
                xyxy = boxes[:, :4].clone()
                xyxy[:, 2] += ALIGNED_FLAG.offset * sw
                xyxy[:, 3] += ALIGNED_FLAG.offset * sh
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, width)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, height)
                x0, y0, x1, y1 = xyxy.unbind(-1)
                normalized = torch.stack(((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                                          (x1 - x0) / width, (y1 - y0) / height), dim=-1)
                targets.append({'boxes': normalized, 'labels': boxes[:, 4].long() - 1})
            losses = self.criterion(predictions, targets)
            return {'DETR.' + name: loss * self.weights[name]
                    for name, loss in losses.items() if name in self.weights}
        sizes = images.new_tensor([[float(info[0]), float(info[1])] for info in information])
        rows = []
        for index, (result, info) in enumerate(zip(self.postprocess(predictions, sizes), information)):
            scale = info[2]
            sh, sw = scale if isinstance(scale, (tuple, list)) else (scale, scale)
            boxes = result['boxes'].clone()
            boxes[:, 2] -= ALIGNED_FLAG.offset * sw
            boxes[:, 3] -= ALIGNED_FLAG.offset * sh
            rows.append(torch.cat((boxes.new_full((len(boxes), 1), index), boxes,
                                   result['scores'][:, None], (result['labels'] + 1).to(boxes)[:, None]), dim=1))
        return {'dt_bboxes': torch.cat(rows, dim=0)}
