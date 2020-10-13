"""
@author: Guanghan Ning
@file: base_trainer.py
@time: 10/9/20 2:43 下午
@file_desc:
"""

from src.trainer.base_worker import Worker
from src.core.class_factory import ClassType, ClassFactory

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
    '''
    optim = OptimConfig  #TODO
    lr_scheduler = LrSchedulerConfig  #TODO
    metric = MetricsConfig  #TODO
    loss = LossConfig   #TODO
    '''
    # Validation
    with_valid = True
    valid_interval = 1

    callbacks = None
    grad_clip = None
    model_statistics = True

@ClassFactory.register(ClassType.TRAINER)
class Trainer(Worker):
    config = TrainerConfig()
