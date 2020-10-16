"""
@author: Guanghan Ning
@file: optimizer.py
@time: 10/13/20 10:24 上午
@file_desc: Optimizer class
"""
import glog as log
from src.core.class_factory import ClassFactory, ClassType
from src.utils.read_configure import class2config
import copy

class Optimizer(object):
    """Register and call Optimizer class."""

    def __init__(self):
        """Initialize."""
        from src.core.default_config import OptimConfig
        self.config = copy.deepcopy(OptimConfig())
        print(self.config)

        print(OptimConfig)
        optim_name = self.config.type
        self.optim_cls = ClassFactory.get_cls(ClassType.OPTIM, optim_name)

    def __call__(self, model=None):
        """Call Optimizer class.

        Arguments:
            model: model, used in torch case

        Return:
            optimizer
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
