"""
@author: Guanghan Ning
@file: model_checkpoint.py
@time: 10/22/20 8:58 下午
@file_desc: Model Checkpoint
"""

import torch
import os, pickle
import glog as log
from src.trainer.base_callback import Callback
from src.core.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class ModelCheckpoint(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(Callback, self).__init__()
        self.priority = 240

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.is_chief = True

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if logs.get('summary_perfs').get('best_valid_perfs_changed', False):
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """Save checkpoint."""
        log.info("Start Save Checkpoint, file_name=%s", self.trainer.checkpoint_file_name)
        checkpoint_file = os.path.join("output",
                                       self.trainer.checkpoint_file_name)
        log.info("Start Save Model, model_file=%s", self.trainer.model_pickle_file_name)

        model_pickle_file = os.path.join("output",
                                         self.trainer.model_pickle_file_name)
        # pickle model
        with open(model_pickle_file, 'wb') as handle:
            pickle.dump(self.trainer.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'weight': self.trainer.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'lr_scheduler': self.trainer.lr_scheduler.state_dict(),
        }
        torch.save(ckpt, checkpoint_file)
        self.trainer.checkpoint_file = checkpoint_file
        self.trainer.model_path = model_pickle_file

    def after_train(self, logs=None):
        """Be called after the training process."""
        torch.save(self.trainer.model.state_dict(), self.trainer.weights_file)
