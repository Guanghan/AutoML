"""
@author: Guanghan Ning
@file: optimizer.py
@time: 10/13/20 10:24 上午
@file_desc: Optimizer class
"""
import glog as log
from src.core.class_factory import ClassFactory, ClassType
from src.utils.read_configure import class2config


class OptimConfig(object):
    """Default Optim Config."""

    _class_type = "trainer.optim"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'Adam'
    params = {"lr": 0.1}


class Optimizer(object):
    """Register and call Optimizer class."""

    config = OptimConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch/tensorflow optim as default
        optim_name = self.config.type
        self.optim_cls = ClassFactory.get_cls(ClassType.OPTIM, optim_name)

    def __call__(self, model=None, lr_scheduler=None, epoch=None, distributed=False):
        """Call Optimizer class.

        :param model: model, used in torch case
        :param lr_scheduler: learning rate scheduler, used in tf case
        :param epoch: epoch of training, used in tf case
        :param distributed: use distributed
        :return: optimizer
        """
        params = class2config(self.config).get("params", {})
        log.info("Calling Optimizer. name={}, params={}".format(self.optim_cls.__name__, params))
        optimizer = None
        try:
            learnable_params = [param for param in model.parameters() if param.requires_grad]
            optimizer = self.optim_cls(learnable_params, **params)
            return optimizer
        except Exception as ex:
            log.error("Failed to call Optimizer name={}, params={}".format(self.optim_cls.__name__, params))
            raise ex


# register all public classes from pytorch's optimizer
import torch.optim as optimizer_package
ClassFactory.register_from_package(optimizer_package, ClassType.OPTIM)
