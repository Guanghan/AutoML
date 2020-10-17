"""
@author: Guanghan Ning
@file: darts.py
@time: 10/5/20 9:31 下午
@file_desc: Implementation of the classic DARTS algorithm
"""
from copy import deepcopy

import glog as log
import os

from src.core.class_factory import ClassFactory, ClassType
from src.search_algorithms.base_algorithm import SearchAlgorithm
from src.search_space.search_space import SearchSpace
from src.trainer.base_callback import Callback
from src.utils.read_configure import desc2config
from src.utils.utils_json import read_json_from_file


@ClassFactory.register(ClassType.CALLBACK)
class DartsTrainer(Callback):
    """A special callback for DartsTrainer."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.device = self.trainer.config.device

        # search algorithm
        self.search_alg = SearchAlgorithm(SearchSpace().search_space)
        self.unrolled = self.trainer.config.unrolled

        # model
        self.model = self.trainer.model
        print("self.model = {}".format(self.trainer.model))
        self._set_algorithm_model(self.model)

        # trainer
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.loss = self.trainer.loss

        # data
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.val_loader = self.trainer._init_dataloader(mode='val')

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        # validation loss for alpha, training loss for weights
        self.val_loader_iter = iter(self.trainer.val_loader)

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""
        # Get current train batch directly from logs
        train_batch = logs['train_batch']
        train_input, train_target = train_batch
        val_input, val_target = next(self.val_loader_iter)
        '''
        try:
            val_input, val_target = next(self.val_loader_iter)
        except Exception:
            self.val_loader_iter = iter(self.trainer.val_loader)
            val_input, val_target = next(self.val_loader_iter)
        '''
        val_input, val_target = val_input.to(self.device), val_target.to(self.device)

        # Call arch search step
        self._train_arch_step(train_input, train_target, val_input, val_target)

    def _train_arch_step(self, train_input, train_target, valid_input, valid_target):
        lr = self.lr_scheduler.get_lr()[0]
        self.search_alg.step(train_input, train_target, valid_input, valid_target,
                             lr, self.optimizer, self.loss, self.unrolled)

    # [train_step] using default from base_trainer, no need to inherit

    # [after_train_step] using default from base_trainer, no need to inherit

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        child_desc_temp = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        log.info('normal = %s', child_desc_temp[0])
        log.info('reduce = %s', child_desc_temp[1])
        self._save_descript()

    def after_train(self, logs=None):
        """Be called after Training."""
        self.trainer._backup()

    def _get_arch_weights(self):
        """Get trained alpha params."""
        return self.model.arch_weights

    def _set_algorithm_model(self, model):
        """ Choose model to search architecture for."""
        self.search_alg.set_model(model)

    def _save_descript(self):
        """Save result description."""
        # get genotypes from trained alpha params (softmax to crisp)
        genotypes = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        # load a template supernet description to modify on
        template_path = os.path.join(os.path.dirname(__file__), "../../src/baselines/baseline_darts.json")
        descript_dict = read_json_from_file(template_path)
        template = desc2config(descript_dict)
        # only replace the genotypes on the template description
        model_desc = self._gen_model_desc(genotypes, template)
        self.trainer.config.codec = model_desc

    def _gen_model_desc(self, genotypes, template):
        """update template supernet description with given genotypes."""
        model_desc = deepcopy(template)
        model_desc.super_network.normal.genotype = genotypes[0]
        model_desc.super_network.reduce.genotype = genotypes[1]
        return model_desc
