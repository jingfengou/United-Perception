import torch
import torch.nn.functional as F
import numpy as np

from .entropy_loss import GeneralizedCrossEntropyLoss
from up.utils.general.registry_factory import LOSSES_REGISTRY
from up.models.losses.loss import _reduce
from up.extensions import SigmoidFocalLossFunction

__all__ = ['SigmoidFocalLoss', 'TorchSigmoidFocalLoss']


@LOSSES_REGISTRY.register('sigmoid_focal_loss')
class SigmoidFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 num_classes,
                 alpha=0.25,
                 gamma=2.0,
                 init_prior=0.01,
                 name='sigmoid_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 dynamic_normalizer=False,
                 ):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - alpha (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
            - dynamic_normalizer (:obj:`bool`): flag of using dynamic_normalizer
        """
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior
        self.num_channels = num_classes - 1
        self.gamma = gamma
        self.alpha = alpha
        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def forward(self, input, target, reduction, normalizer=None, weights=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        # assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1).int()
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(input, target, self.alpha, self.gamma)
        if weights is not None:
            if weights.shape.__len__() == 2 or (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
                loss = SigmoidFocalLossFunction.apply(
                    input,
                    target,
                    normalizer,
                    self.gamma,
                    self.alpha,
                    self.num_channels,
                    'none')
                weights = weights.reshape(-1).unsqueeze(-1)
                assert weights.shape.__len__() == loss.shape.__len__()
                loss = loss * weights
                return _reduce(loss, reduction='sum')

        loss = SigmoidFocalLossFunction.apply(
            input, target, normalizer, self.gamma, self.alpha, self.num_channels, reduction)
        return loss


@LOSSES_REGISTRY.register('torch_sigmoid_focal_loss')
class TorchSigmoidFocalLoss(GeneralizedCrossEntropyLoss):
    """
    Sigmoid focal loss implemented with PyTorch operators.
    """

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 init_prior=0.01,
                 name='torch_sigmoid_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 dynamic_normalizer=False,
                 use_sigmoid=True):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
            - dynamic_normalizer (:obj:`bool`): flag of using dynamic_normalizer
        """
        self.use_sigmoid = use_sigmoid
        activation_type = 'sigmoid' if use_sigmoid else 'softmax'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior
        self.gamma = gamma
        self.alpha = alpha
        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def generate_onehot_target(self, input, target):
        # N * C
        N, C = input.size()
        onehot_target = input.new_zeros(N, C + 1)
        onehot_target[torch.arange(N), target] = 1
        return onehot_target[:, 1:]

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1)
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(input, target.int(), self.alpha, self.gamma)
        sample_mask = torch.nonzero(target != self.ignore_index).reshape(-1)
        if sample_mask.size(0) != target.size(0):
            input = input[sample_mask, :]
            target = target[sample_mask]
        target = self.generate_onehot_target(input, target)
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if isinstance(self.alpha, list):
            alpha_t = torch.from_numpy(np.array(self.alpha[1:])).to(target.device)
            alpha_t = alpha_t.view(-1, len(self.alpha) - 1).expand_as(target)
            alpha_t = alpha_t * target + (1 - self.alpha[0]) * (1 - target)
            loss = alpha_t * loss
        elif self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        else:
            pass
        return _reduce(loss, reduction=reduction, normalizer=normalizer)


def dynamic_normalizer(input, target, alpha, gamma):
    def reduce_(tensor, gamma):
        return tensor.pow(gamma).sum()
    target = target.reshape(-1).long()
    input_p = input.detach().sigmoid()
    pos_mask = torch.nonzero(target >= 1).squeeze()
    valid_mask = torch.nonzero(target >= 0).squeeze()
    pos_normalizer = reduce_((1 - input_p[pos_mask, target[pos_mask] - 1]), gamma)
    neg_normalizer = reduce_(input_p[valid_mask], gamma) - reduce_(input_p[pos_mask, target[pos_mask] - 1], gamma)
    pos_normalizer *= alpha
    neg_normalizer *= 1 - alpha
    normalizer = torch.clamp(pos_normalizer + neg_normalizer, min=1)
    return normalizer
