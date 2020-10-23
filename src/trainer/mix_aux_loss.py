"""
@author: Guanghan Ning
@file: mix_aux_loss.py
@time: 10/23/20 11:23 上午
@file_desc:
"""
import torch.nn as nn
from src.core.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class MixAuxiliaryLoss(nn.Module):
    """Class of Mix Auxiliary Loss.

    :param aux_weight: auxiliary loss weight
    :type aux_weight: float
    :loss_base: base loss function
    :loss_base: str
    """

    def __init__(self, aux_weight, loss_base):
        """Init MixAuxiliaryLoss."""
        super(MixAuxiliaryLoss, self).__init__()
        self.aux_weight = aux_weight
        loss_base_cp = loss_base.copy()
        loss_base_name = loss_base_cp.pop('type')
        self.loss_fn = eval(loss_base_name)(**loss_base_cp)

    def forward(self, outputs, targets):
        """Loss forward function."""
        if len(outputs) != 2:
            raise Exception('outputs length must be 2')
        loss0 = self.loss_fn(outputs[0], targets)
        loss1 = self.loss_fn(outputs[1], targets)
        return loss0 + self.aux_weight * loss1
