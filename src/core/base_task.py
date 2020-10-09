"""
@author: Guanghan Ning
@file: base_task.py
@time: 10/9/20 10:26 上午
@file_desc: The base class for tasks, e.g., algorithms, datasets, etc.
"""
class Task(object):
    def __init__(self, task_id=None, worker_id=None):
        """Init Task class"""
