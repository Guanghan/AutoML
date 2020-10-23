"""
@author: Guanghan Ning
@file: base_loss.py
@time: 10/13/20 10:13
@file_desc: Loss class
"""
import glog as log
from inspect import isclass
from functools import partial
from src.core.class_factory import ClassFactory, ClassType
from src.core.default_config import LossConfig
from src.utils.utils_cfg import class2config, Config
from src.core.default_config import TrainerConfig


class Loss(object):
    """Register and call loss class."""

    config = LossConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch loss as default
        loss_name = self.config.type
        self._cls = ClassFactory.get_cls(ClassType.LOSS, loss_name)

    def __call__(self):
        """Call loss cls."""
        #params = class2config(self.config).get("params", {})
        dst_config = Config()
        class2config(config_dst=dst_config, class_src=self.config)
        params = dst_config.get("params", {})
        log.info("Call Loss. name={}, params={}".format(self._cls.__name__, params))
        try:
            if params:
                cls_obj = self._cls(**params) if isclass(self._cls) else partial(self._cls, **params)
            else:
                cls_obj = self._cls() if isclass(self._cls) else partial(self._cls)
            if TrainerConfig().cuda:
                cls_obj = cls_obj.cuda()
            return cls_obj
        except Exception as ex:
            log.error("Failed to call Loss name={}, params={}".format(self._cls.__name__, params))
            raise ex


# register all public loss classes from pytorch's basic neural network building blocks
import torch.nn as nn_package
ClassFactory.register_from_package(nn_package, ClassType.LOSS)

# register all public classes from timm's loss module
try:
    # PyTorch Image Models (timm) is a collection of image models,
    # layers, utilities, optimizers, schedulers, data-loaders / augmentations,
    # and reference training / validation scripts that aim to pull together
    # a wide variety of SOTA models with ability to reproduce ImageNet training results.
    # https://github.com/rwightman/pytorch-image-models
    import timm.loss as timm_loss
    ClassFactory.register_from_package(timm_loss, ClassType.LOSS)
except Exception as ex:
    log.warn("timm not been installed but it is okay, {}".format(str(ex)))
    pass



