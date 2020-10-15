"""
@author: Guanghan Ning
@file: base_pipestep.py
@time: 10/14/20 8:48 下午
@file_desc: Basic Step in a whole Pipeline
"""
from src.core.base_task import Task
from src.core.class_factory import ClassFactory, ClassType

from src.dataset.base_dataset import DatasetConfig
from src.search_algorithms.base_algorithm import SearchAlgorithmConfig
from src.search_space.search_space import SearchSpaceConfig, ModelConfig
from src.trainer.base_trainer import TrainerConfig


class PipeStepConfig(object):
    """Default Pipeline config for Pipe Step."""

    dataset = DatasetConfig
    search_algorithm = SearchAlgorithmConfig
    search_space = SearchSpaceConfig
    model = ModelConfig
    trainer = TrainerConfig
    #evaluator = EvaluatorConfig  #TODO
    pipe_step = {}


class PipeStep(object):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        self.task = Task()

    def __new__(cls):
        """Create pipe step instance by ClassFactory."""
        t_cls = ClassFactory.get_cls(ClassType.PIPE_STEP)
        return super().__new__(t_cls)

    def run(self):
        """Conduct the main task in this pipe step."""
        raise NotImplementedError
