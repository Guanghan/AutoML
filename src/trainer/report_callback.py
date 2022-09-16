"""
@author: Guanghan Ning
@file: report_callback.py
@time: 10/22/20 10:48 下午
@file_desc:
"""
import logging
from src.trainer.base_callback import Callback
from src.utils.utils_saver import Saver
from src.core.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class ReportCallback(Callback):
    """Callback that report records."""

    def __init__(self):
        """Initialize ReportCallback callback."""
        super(Callback, self).__init__()
        self.epoch = 0
        self.priority = 280

    def after_valid(self, logs=None):
        """Be called after each epoch."""
        self._save()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.epoch = epoch
        self._save(epoch)

    def after_train(self, logs=None):
        """Close the connection of report."""
        self._save(self.epoch)

    def _save(self, epoch=None):
        record = Saver().receive(self.trainer.step_name, self.trainer.worker_id)
        record.epoch = epoch
        if self.trainer.config.codec:
            record.desc = self.trainer.config.codec
        if not record.desc:
            record.desc = self.trainer.model_desc
        record.performance = self.trainer.performance
        record.objectives = self.trainer.valid_metrics.objectives
        if record.performance is not None:
            for key in record.performance:
                if key not in record.objectives:
                    if (key == 'gflops' or key == 'kparams'):
                        record.objectives.update({key: 'MIN'})
                    else:
                        record.objectives.update({key: 'MAX'})
        record.model_path = self.trainer.model_path
        record.checkpoint_path = self.trainer.checkpoint_file
        record.weights_file = self.trainer.weights_file
        Saver()._save(record)
        logging.debug("report_callback record: {}".format(record))
