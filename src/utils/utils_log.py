"""
@author: Guanghan Ning
@file: utils_log.py
@time: 10/14/20 10:06 上午
@file_desc: logging related utilities
"""

import os, sys
from datetime import datetime

import glog as log
from src.utils.utils_io_folder import create_folder


class TaskConfig(dict):
    """Task Config."""

    task_id = datetime.now().strftime('%m%d.%H%M%S.%f')[:-3]
    local_base_path = "./tasks"
    output_subpath = "output"
    best_model_subpath = "best_model"
    log_subpath = "logs"
    result_subpath = "result"
    worker_subpath = "workers/[step_name]/[worker_id]"
    backup_base_path = None
    use_dloop = False


class General(object):
    """General Config."""

    task = TaskConfig
    step_name = None
    worker_id = None
    backend = 'pytorch'
    device_category = 'GPU'
    env = None
    calc_params_each_epoch = False


def init_log(level='info', log_file="log.txt"):
    """Init logging configuration."""
    log_path = "./logs/"
    create_folder(log_path)
    if level == "info":
        log.setLevel("INFO")
    elif level == "warn":
        log.setLevel("WARNING")
    elif level == "error":
        log.setLevel("ERROR")
    elif level == "fatal":
        log.setLevel("FATAL")
    else:
        raise ("Not supported logging level: {}".format(level))
    return log

