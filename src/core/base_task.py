"""
@author: Guanghan Ning
@file: base_task.py
@time: 10/9/20 10:26
@file_desc: The base class for tasks, e.g., algorithms, datasets, etc.

TaskClass

tasks/[task_id]/
    +-- output
    |       +-- [step name]
    |               +-- model_desc.json          # model description, save_model_desc(model_desc, performance)
    |               +-- hyperparameters.json     # hyper-parameters, save_hps(hps), load_hps()
    |               +-- models                   # folder, save models, save_model(id, model, desc, performance)
    |                     +-- (worker_id).pth    # model file
    |                     +-- (worker_id).json   # model desc and performance file
    +-- workers
    |       +-- [step name]
    |       |       +-- [worker_id]
    |       |       +-- [worker_id]
    |       |               +-- logs
    |       |               +-- checkpoints
    |       |               |     +-- [epoch num].pth
    |       |               |     +-- model.pkl
    |       |               +-- result
    |       +-- [step name]
    +-- logs
"""

import os
from datetime import datetime
from src.utils.utils_io_folder import create_folder
from src.core.default_config import TaskConfig, GeneralConfig


class Task(object):
    def __init__(self, task_id=None, step_name = None, worker_id=None):
        """Init Task class"""
        self.task_cfg = TaskConfig()
        self.task_id = task_id if task_id is not None else self.task_cfg.task_id
        self.step_name = step_name if step_name is not None else GeneralConfig.step_name

    @property
    def task_id_property(self):
        """Property: task_id."""
        return self.task_id

    @property
    def base_path(self):
        _base_path = self.task_cfg.base_path
        create_folder(_base_path)
        return _base_path

    @property
    def log_path(self):
        _log_path = os.path.join(self.task_cfg.base_path,
                                 self.task_cfg.log_folder)
        create_folder(_log_path)
        return _log_path

    @property
    def output_path(self):
        _output_path = os.path.join(self.task_cfg.base_path,
                                    self.task_cfg.output_folder)
        create_folder(_output_path)
        return _output_path

    @property
    def best_model_path(self):
        _best_model_path = os.path.join(self.task_cfg.base_path,
                                        self.task_cfg.best_model_folder)
        create_folder(_best_model_path)
        return _best_model_path
