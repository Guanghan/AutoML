"""
@author: Guanghan Ning
@file: pipeline.py
@time: 10/2/20 6:55
@file_desc: The AutoML pipeline
"""

import glog as log

log.setLevel('INFO')

from src.utils.utils_cfg import Config, desc2config
from src.core.class_factory import ClassFactory, ClassType
from src.core.default_config import PipeStepConfig


class Pipeline(object):
    """
    The AutoML pipeline:
    1. load configurations from yaml config file
    2. for each step in the pipeline:
       attach corresponding configurations to ClassFactory,
    3. for each configuration:
       find the corresponding class to use from registered classes
       Note: each implemented class is pre-registered with this decorator: @ClassFactory.register(ClassType.$TYPE)
    """

    def __init__(self, config_path):
        """ Initialize the pipeline

        Args:
            config_path: full path to config file
        """
        # register customized classes for usage
        self.__register__()

        # load config file (to choose particular class)
        log.info("Loading configuration from file: {}".format(config_path))
        self.config = Config(config_path)

    def __register__(self):
        """ Register customized classes
            Example 1:
            @ClassFactory.register(ClassType.SEARCH_ALGORITHM)
            class my_NAS(object):
                def run(self):
                    print("Customized neural architecture search algorithm.")

            Example 2:
            from src.dataset.cifar10 import Cifar10
            ClassFactory.register(Cifar10, ClassType.DATASET)
        """
        pass

    def run(self):
        """ Run the pipeline

        """
        ClassFactory.attach_config_to_factory(self.config)

        #procedures = ["nas"]
        procedures = ["nas", "fully_train"]

        for procedure in procedures:
            # get configuration for each step
            log.info("Select configuration for procedure: {}".format(procedure))
            step_cfg = self.config.get(procedure)

            # set current step's config to ClassFactory [class into __registry__]
            log.info("Set {} config to ClassFactory [class into __registry__]".format(procedure))
            ClassFactory().attach_config_to_factory(step_cfg)

            # load Config from current step description [description into __config__]
            log.info("load config from {} description [description into __config__]".format(procedure))
            desc2config(config_dst=PipeStepConfig, desc_src=step_cfg)

            # get corresponding class given an attribute
            log.info("__config__ = {}\n".format(ClassFactory.__configs__))
            log.info("__registry__ = {}\n".format(ClassFactory.__registry__))
            step_cls = ClassFactory.get_cls(ClassType.PIPE_STEP)

            # instantiate corresponding class
            step = step_cls()

            # run step
            log.info("Running step: {}".format(type(step).__name__))
            step.run()


if __name__ == '__main__':
    #pipeline = Pipeline('../../configs/darts.yaml')
    #pipeline = Pipeline('configs/darts.yaml')
    pipeline = Pipeline('../../configs/darts_full.yaml')
    pipeline.run()
