"""
@author: Guanghan Ning
@file: pipeline.py
@time: 10/2/20 6:55 下午
@file_desc: The AutoML pipeline
"""

import glog as logger

logger.setLevel('INFO')

from src.utils.utils_cfg import Config
from src.core.class_factory import ClassFactory, ClassType


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
        # register available classes for usage
        self.__register__()

        # load config file (to choose particular class)
        self.config = Config(config_path)

    def __register__(self):
        """ Register available classes

        """
        @ClassFactory.register(ClassType.SEARCH_ALGORITHM)
        class my_NAS(object):
            def run(self):
                print("Running neural architecture search algorithm.")

        @ClassFactory.register(ClassType.SEARCH_SPACE)
        class my_space(object):
            def run(self):
                print("Running space search definition")

        @ClassFactory.register(ClassType.DATASET)
        class COCO(object):
            def run(self):
                print("Running COCO dataset initialization")

        @ClassFactory.register(ClassType.DATASET)
        class CIFAR(object):
            def run(self):
                print("Running CIFAR dataset initialization")

    def run(self):
        """ Run the pipeline

        """
        # attach config to ClassFactory
        ClassFactory.attach_config_to_factory(self.config)

        procedures = ['search_algorithm', 'search_space', 'dataset']
        for procedure in procedures:
            # get corresponding class given an attribute
            step_cls = ClassFactory.get_cls(procedure)

            # instantiate corresponding class
            step = step_cls()

            # run step
            step.run()


if __name__ == '__main__':
    pipeline = Pipeline('../../configs/example.yaml')
    pipeline.run()
