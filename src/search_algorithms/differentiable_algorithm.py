"""
@author: Guanghan Ning
@file: differentiable_algorithm.py
@time: 10/8/20 11:33
@file_desc: Differentiable gradient method for neural architecture search (first proposed in DARTS)
"""

import importlib
from src.core.class_factory import ClassType, ClassFactory
from src.search_algorithms.base_algorithm import SearchAlgorithm
from src.core.default_config import TrainerConfig
from src.search_space.description import NetworkDesc


class DifferentialConfig(object):
    """Config for Differential."""

    sample_num = 1
    momentum = 0.9
    weight_decay = 3.0e-4
    parallel = False
    codec = 'DartsCodec'
    arch_optim = dict(type='Adam', lr=3.0e-4, betas=[0.5, 0.999], weight_decay=1.0e-3)
    criterion = dict(type='CrossEntropyLoss')
    tf_arch_optim = dict(type='AdamOptimizer', learning_rate=3.0e-4, beta1=0.5, beta2=0.999)
    tf_criterion = dict(type='CrossEntropyWeightDecay', cross_entropy='sparse_softmax_cross_entropy',
                        weight_decay=1.0e-3)
    objective_keys = 'accuracy'


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DifferentialAlgorithm(SearchAlgorithm):
    """Differential algorithm.

    Args:
        search_space (SearchSpace): Input search_space.
    """

    config = DifferentialConfig()
    trainer_config = TrainerConfig()

    def __init__(self, search_space=None):
        """Init DifferentialAlgorithm."""
        super(DifferentialAlgorithm, self).__init__(search_space)
        self.network_momentum = self.config.momentum
        self.network_weight_decay = self.config.weight_decay
        self.parallel = self.config.parallel
        self.criterion = self.config.criterion
        self.sample_num = self.config.sample_num
        self.sample_idx = 0

    def new_model(self):
        """Build new model from Network description of search space."""
        net_desc = NetworkDesc(self.search_space)
        model_new = net_desc.to_model().cuda()
        for x, y in zip(model_new.arch_parameters(), self.model.arch_parameters()):
            x.detach().copy_(y.detach())
        return model_new

    def set_model(self, model):
        """Set with existing model."""
        self.model = model.module if self.parallel else model
        self.loss = self._init_loss().cuda()
        self.optimizer = self._init_arch_optimizer(self.model)

    def _init_arch_optimizer(self, model=None):
        """Init arch optimizer."""
        optim_config = self.config.arch_optim.copy()
        optim_name = optim_config.pop('type')
        optim_class = getattr(importlib.import_module('torch.optim'),
                              optim_name)
        learnable_params = model.arch_parameters()
        optimizer = optim_class(learnable_params, **optim_config)
        return optimizer

    def _init_loss(self):
        """Init loss."""
        loss_config = self.criterion.copy()
        loss_name = loss_config.pop('type')
        loss_class = getattr(importlib.import_module('torch.nn'), loss_name)
        return loss_class(**loss_config)

    def step(self, train_x=None, train_y=None, valid_x=None, valid_y=None,
             lr=None, w_optimizer=None, w_loss=None, unrolled=None, scope_name=None):
        """Compute one step."""
        self.optimizer.zero_grad()
        loss = w_loss(self.model(valid_x), valid_y)
        loss.backward()
        self.optimizer.step()
        return

    def search(self):
        """Search function."""
        return self.sample_idx, self.search_space

    @property
    def is_completed(self):
        """Check if the search is finished."""
        self.sample_idx += 1
        return self.sample_idx > self.sample_num

