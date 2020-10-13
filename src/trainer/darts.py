"""
@author: Guanghan Ning
@file: darts.py
@time: 10/5/20 9:31 下午
@file_desc: Implementation of the classic DARTS algorithm
"""

from src.core.class_factory import ClassFactory, ClassType
from src.trainer.base_callback import Callback
from src.search_algorithms.base_algorithm import SearchAlgorithm
from src.search_space.search_space import SearchSpace

@ClassFactory.register(ClassType.CALLBACK)
class DartsTrainer(Callback):
    """A special callback for DartsTrainer."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.search_alg = SearchAlgorithm(SearchSpace().search_space)
        self._set_algorithm_model(self.model)

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""

    def after_train(self, logs=None):
        """Be called after Training."""

    def _get_arch_weights(self):
        """Save result descript."""

    def _save_descript(self):
        """Save result descript."""

    def _gen_model_desc(self, genotypes, template):
        """Save result descript."""
