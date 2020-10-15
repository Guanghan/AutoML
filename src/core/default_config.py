"""
@author: Guanghan Ning
@file: default_config.py
@time: 10/15/20 1:14 下午
@file_desc: Default Config for all classes, to prevent circular dependent imports
"""

from src.core.class_factory import ClassType, ClassFactory




class DatasetConfig(object):
    """Default Dataset config for Pipeline."""

    type = "Cifar10"
    _class_type = ClassType.DATASET


class BaseConfig(object):
    """Base config of dataset."""

    data_path = None
    batch_size = 1
    num_workers = 0
    imgs_per_gpu = 1,
    shuffle = False
    download = False
    pin_memory = True
    drop_last = True
    transforms = []

class SearchAlgorithmConfig(object):
    """Default Search Algorithm config for Pipeline."""

    _class_type = ClassType.SEARCH_ALGORITHM
    type = None


class SearchSpaceConfig(object):
    """Default Search Space config for Pipeline."""

    _type_name = ClassType.SEARCH_SPACE
    type = None


class ModelConfig(object):
    """Default Model config for Pipeline."""

    _type_name = ClassType.SEARCH_SPACE
    type = None
    model_desc = None
    model_desc_file = None
    model_file = None


class LrSchedulerConfig(object):
    """Default LrScheduler Config."""

    _class_type = "trainer.lr_scheduler"
    _update_all_attrs = True
    _exclude_keys = ['type']
    type = 'MultiStepLR'
    params = {"milestones": [75, 150], "gamma": 0.5}


class MetricsConfig(object):
    """Default Metrics Config."""

    _class_type = "trainer.metric"
    _update_all_attrs = True
    type = 'accuracy'
    params = {}


class OptimConfig(object):
    """Default Optim Config."""

    _class_type = "trainer.optim"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'Adam'
    params = {"lr": 0.1}


class LossConfig(object):
    """Default Loss Config."""

    _class_type = "trainer.loss"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'CrossEntropyLoss'
    params = {}


class TrainerConfig(object):
    """Default Trainer Config."""
    # GPU
    cuda = True
    device = cuda if cuda is not True else 0
    # Model
    pretrained_model_file = None
    save_model_desc = False
    # Report
    report_freq = 10
    # Training
    seed = 0
    epochs = 1
    optim = OptimConfig
    lr_scheduler = LrSchedulerConfig
    metric = MetricsConfig
    loss = LossConfig
    # Validation
    with_valid = True
    valid_interval = 1

    callbacks = None
    grad_clip = None
    model_statistics = True


class PipeStepConfig(object):
    """Default Pipeline config for Pipe Step."""

    dataset = DatasetConfig
    search_algorithm = SearchAlgorithmConfig
    search_space = SearchSpaceConfig
    model = ModelConfig
    trainer = TrainerConfig
    #evaluator = EvaluatorConfig  #TODO
    pipe_step = {}
