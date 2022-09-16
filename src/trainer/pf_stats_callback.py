"""
@author: Guanghan Ning
@file: pf_stats_callback.py
@time: 10/22/20 8:50 下午
@file_desc: Statistcs for Params and Flops.
"""

import logging
import torch

from src.core.class_factory import ClassFactory, ClassType
from src.trainer.base_callback import Callback
from src.utils.utils_flops import calc_model_flops_params


@ClassFactory.register(ClassType.CALLBACK)
class ModelStatistics(Callback):
    """Callback that log statistics about model after each epoch."""

    def __init__(self):
        """Initialize ModelStatistics callback."""
        super(Callback, self).__init__()
        self.priority = 220

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.input = None
        self.gflops = None
        self.kparams = None
        self.calc_params_each_epoch = True

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch of Training."""
        try:
            if self.input is None:
                input, target = logs['train_batch']
                self.input = torch.unsqueeze(input[0], 0)
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if self.calc_params_each_epoch:
            self.update_flops_params(epoch=epoch, logs=logs)

    def after_train(self, logs=None):
        """Be called after train."""
        if not self.calc_params_each_epoch:
            self.update_flops_params(logs=logs)

    def update_flops_params(self, epoch=None, logs=None):
        """Calculate flops and params."""
        self.model = self.trainer.model
        try:
            if self.gflops is None:
                flops_count, params_count = calc_model_flops_params(self.model,
                                                                    self.input)
                #self.gflops, self.kparams = flops_count * 1600 * 1e-9, params_count * 1e-3
                self.gflops, self.kparams = flops_count * 1e-9, params_count * 1e-3
            summary_perfs = logs.get('summary_perfs', {})
            if epoch:
                summary_perfs.update({'gflops': self.gflops, 'kparams': self.kparams,
                                      'epoch': epoch})
            else:
                summary_perfs.update({'gflops': self.gflops, 'kparams': self.kparams})
            logs.update({'summary_perfs': summary_perfs})
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)

