"""
@author: Guanghan Ning
@file: lr_scheduler.py
@time: 10/13/20 10:01
@file_desc: Learning rate scheduler class.
"""
import glog as log
from src.core.class_factory import ClassFactory, ClassType
from src.core.default_config import LrSchedulerConfig
from src.utils.utils_cfg import class2config, Config


class DefaultLrSchedulerConfig(object):
    """Default LrScheduler Config."""

    _class_type = "trainer.lr_scheduler"
    _update_all_attrs = True
    _exclude_keys = ['type']
    type = 'MultiStepLR'
    params = {"milestones": [75, 150], "gamma": 0.5}


class LrScheduler(object):
    """Register and call LrScheduler class."""

    #config = LrSchedulerConfig()
    config = DefaultLrSchedulerConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch's optim as default
        self._cls = ClassFactory.get_cls(ClassType.LR_SCHEDULER, self.config.type)

    def __call__(self, optimizer=None, epochs=None, steps=None):
        """Call lr scheduler class."""
        #params = class2config(self.config).get("params", {})
        dst_config = class2config(Config(), self.config)
        params = dst_config.get("params", {})
        log.info("Calling LrScheduler. name={}, params={}".format(self._cls.__name__, params))
        try:
            if params and optimizer:
                return self._cls(optimizer, **params)
            elif optimizer:
                return self._cls(optimizer)
            else:
                return self._cls(**params)
        except Exception as ex:
            log.error("Failed to call LrScheduler name={}, params={}".format(self._cls.__name__, params))
            raise ex


# register all public classes from pytorch's lr_scheduler
import torch.optim.lr_scheduler as lr_scheduler_package
ClassFactory.register_from_package(lr_scheduler_package, ClassType.LR_SCHEDULER)

