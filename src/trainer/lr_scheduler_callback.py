"""
@author: Guanghan Ning
@file: lr_scheduler_callback.py
@time: 10/16/20 1:59 上午
@file_desc: LearningRateScheduler callback.
"""

from .base_callback import Callback
from src.core.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class LearningRateScheduler(Callback):
    """Callback that adjust the learning rate during training."""

    def __init__(self):
        """Initialize LearningRateScheduler callback."""
        super(Callback, self).__init__()
        self.priority = 260

    def before_train(self, logs=None):
        """Be called before training."""
        #self.call_point = self.trainer.config.lr_adjustment_position
        self.call_point = "after_epoch"
        self.lr_scheduler = self.trainer.lr_scheduler

    def before_epoch(self, epoch, logs=None):
        """Call before_epoch of the managed callbacks."""
        self.epoch = epoch

    def after_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        if self.call_point.upper() == "AFTER_EPOCH" and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch=epoch)

    def before_train_step(self, batch_index, logs=None):
        """Call before_train_step of the managed callbacks."""
        if self.call_point.upper() == 'BEFORE_TRAIN_STEP':
            self.lr_scheduler.step(epoch=self.epoch)

    def after_train_step(self, batch_index, logs=None):
        """Call after_train_step of the managed callbacks."""
        if self.call_point.upper() == 'AFTER_TRAIN_STEP':
            self.lr_scheduler.step(epoch=self.epoch)

