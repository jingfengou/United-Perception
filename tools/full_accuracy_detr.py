"""Full-COCO official DETR, identical external adapter in all three UP lanes.

Original UP has no DETR. The pinned official transformer, matching and loss
are unmodified. Only dataset/result conventions are translated at this boundary.
Training starts with an ImageNet backbone, not a COCO-trained detector.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from accuracy_detr import REVISION, target_boxes
from accuracy_data import digest
from full_accuracy_data import COCO_CATEGORY_IDS


class FullCocoDetrForUP(nn.Module):
    def __init__(self, source, source_sha256, checkpoint, checkpoint_sha256, revision):
        super().__init__()
        source = Path(source).resolve()
        files = {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob('*.py'))}
        if (revision != REVISION or digest(checkpoint) != checkpoint_sha256 or
                hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != source_sha256):
            raise ValueError('Pinned official DETR source/ImageNet weights changed')
        sys.path.insert(0, str(source))
        detr = importlib.import_module('models.detr')
        backbone = importlib.import_module('models.backbone')
        transformer = importlib.import_module('models.transformer')
        position = importlib.import_module('models.position_encoding')
        matcher = importlib.import_module('models.matcher')
        misc = importlib.import_module('util.misc')
        if Path(detr.__file__).resolve().parent != source/'models':
            raise RuntimeError('A different official DETR package was imported')
        from torchvision.models import resnet50
        body = resnet50(weights=None, replace_stride_with_dilation=[False, False, False],
                        norm_layer=backbone.FrozenBatchNorm2d)
        body.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        features = backbone.BackboneBase(body, train_backbone=True, num_channels=2048, return_interm_layers=False)
        features = backbone.Joiner(features, position.PositionEmbeddingSine(128, normalize=True))
        features.num_channels = 2048
        self.detector = detr.DETR(features, transformer.Transformer(d_model=256, nhead=8,
            num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=.1,
            return_intermediate_dec=True), num_classes=91, num_queries=100, aux_loss=True)
        self.weights = {'loss_ce': 1., 'loss_bbox': 5., 'loss_giou': 2.}
        self.weights.update({name+'_'+str(index): value for index in range(5)
                            for name, value in list(self.weights.items())})
        self.criterion = detr.SetCriterion(91, matcher.HungarianMatcher(1, 5, 2), self.weights,
                                          .1, ['labels', 'boxes', 'cardinality'])
        self.postprocess = detr.PostProcess()
        self.nested_tensor = misc.NestedTensor
        # Buffers are derived, non-persistent; checkpoint keys follow official DETR.
        self.register_buffer('to_category', torch.tensor((0, *COCO_CATEGORY_IDS)), persistent=False)
        inverse = torch.zeros(91, dtype=torch.long)
        inverse[self.to_category[1:]] = torch.arange(1, 81)
        self.register_buffer('to_up_class', inverse, persistent=False)

    def forward(self, batch):
        from up.utils.general.global_flag import ALIGNED_FLAG
        images, information = batch['image'], batch['image_info']
        mask = images.new_ones((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool)
        for index, info in enumerate(information):
            if float(info[6]) or float(info[7]):
                raise ValueError('DETR data adapter requires no left/top padding')
            mask[index, :int(info[0]), :int(info[1])] = False
        predictions = self.detector(self.nested_tensor(images, mask))
        if self.training:
            targets = []
            for boxes, info in zip(batch['gt_bboxes'], information):
                selected = boxes[boxes[:, 4] > 0]
                labels = self.to_category[selected[:, 4].to(torch.long)]
                targets.append({'boxes': target_boxes(selected[:, :4], info, ALIGNED_FLAG.offset),
                                'labels': labels})
            losses = self.criterion(predictions, targets)
            return {'DETR.'+name: value*self.weights[name] for name, value in losses.items() if name in self.weights}
        sizes = images.new_tensor([[float(info[0]), float(info[1])] for info in information])
        rows = []
        for index, (result, info) in enumerate(zip(self.postprocess(predictions, sizes), information)):
            scale = info[2]
            sh, sw = map(float, scale) if isinstance(scale, (list, tuple)) else (float(scale), float(scale))
            boxes = result['boxes'].clone()
            boxes[:, 2] -= ALIGNED_FLAG.offset*sw
            boxes[:, 3] -= ALIGNED_FLAG.offset*sh
            labels = self.to_up_class[result['labels']]
            # Unused official COCO class slots are ignored by COCOeval. Do so
            # before translating to UP's contiguous 1..80 convention.
            keep = labels > 0
            rows.append(torch.cat((boxes.new_full((len(boxes), 1), index), boxes, result['scores'][:, None],
                                   labels.to(boxes.dtype)[:, None]), dim=1)[keep])
        return {'dt_bboxes': torch.cat(rows, dim=0)}

