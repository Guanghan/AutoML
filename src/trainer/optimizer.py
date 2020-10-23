"""
@author: Guanghan Ning
@file: optimizer.py
@time: 10/13/20 10:24
@file_desc: Optimizer class
"""
import glog as log
from src.core.class_factory import ClassFactory, ClassType
from src.utils.utils_cfg import class2config, Config

from src.core.default_config import OptimConfig
assert OptimConfig.params == {'lr': 0.1}


class DefaultOptimConfig(object):
    _class_type = "trainer.optim"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'Adam'
    params = {"lr": 0.1}


class Optimizer(object):
    """Register and call Optimizer class."""

    #config = OptimConfig()
    config = DefaultOptimConfig()

    def __init__(self):
        """Initialize."""
        optim_name = self.config.type
        self.optim_cls = ClassFactory.get_cls(ClassType.OPTIM, optim_name)

    def __call__(self, model=None):
        """Call Optimizer class.

        Arguments:
            model: model, used in torch case

        Return:
            optimizer
        """
        #params = class2config(self.config).get("params", {})
        dst_config = class2config(Config(), self.config)
        params = dst_config.get("params", {})

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
print("loaded SGD")