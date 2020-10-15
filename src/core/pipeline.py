"""
@author: Guanghan Ning
@file: pipeline.py
@time: 10/2/20 6:55 下午
@file_desc: The AutoML pipeline
"""

import glog as logger

logger.setLevel('INFO')

from src.utils.read_configure import Config, desc2config
from src.core.class_factory import ClassFactory, ClassType
from src.core.base_pipestep import PipeStepConfig


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
        self.config = Config(config_path)

    def __register__(self):
        """ Register customized classes

            @ClassFactory.register(ClassType.SEARCH_ALGORITHM)
            class my_NAS(object):
                def run(self):
                    print("Customized neural architecture search algorithm.")
        """
        pass

    def run(self):
        """ Run the pipeline

        """
        # attach overall config to ClassFactory
        ClassFactory.attach_config_to_factory(self.config)

        procedures = ["nas"]
        for procedure in procedures:
            # get configuration for each step
            step_cfg = self.config.get(procedure)

            # set current step's config to ClassFactory
            ClassFactory().attach_config_to_factory(step_cfg)

            # load Config form current step description
            desc2config(config_dst=PipeStepConfig, desc_src=step_cfg)

            # get corresponding class given an attribute
            step_cls = ClassFactory.get_cls(ClassType.PIPE_STEP)

            # instantiate corresponding class
            step = step_cls()

            # run step
            step.run()


if __name__ == '__main__':
    pipeline = Pipeline('../../configs/darts.yaml')
    pipeline.run()
