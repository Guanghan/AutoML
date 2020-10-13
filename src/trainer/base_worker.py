"""
@author: Guanghan Ning
@file: base_worker.py
@time: 10/9/20 7:26 下午
@file_desc: Base worker for training and evaluating.

It is the basic class of other workers, e.g., TrainWorker and EvaluatorWork.
It loads the pickle file into worker from master, and run the train_process
function of each distributed worker on local node, it also has the function
of timeout, killing the worker process which exceeds setting time.
"""
import copy
from src.core.base_task import Task
from src.core.class_factory import ClassFactory
from src.utils.read_configure import class2config


class Worker(Task):
    """Class of Worker.

    This is a worker used to load worker's pickle file,
    and run the process of training and evaluating.

    Arguments:
        args (dict or Config, default is None): arguments from user config file
    """

    # original params
    __worker_path__ = None
    __worker_module__ = None
    __worker_name__ = None
    # id params
    __worker_id__ = 0
    __config__ = None
    __general__ = None

    def __init__(self, args=None):
        """Init DistributedWorker."""
        super().__init__()
        # privates
        Worker.__worker_id__ += 1
        self._worker_id = Worker.__worker_id__
        self.__env_config__ = copy.deepcopy(ClassFactory.__configs__)
        self.__network_config__ = copy.deepcopy(ClassFactory.__registry__)
        return

    @property
    def worker_id(self):
        """Property: worker_id."""
        return self._worker_id

    @worker_id.setter
    def worker_id(self, value):
        """Setter: set worker_id with value.

        :param value: worker id
        :type value: int
        """
        self._worker_id = value

    def __call__(self):
        """Call function based on GPU devices."""
        raise NotImplementedError

    def train_process(self):
        """Abstract base function for DistributedWorker to do the train process."""
        raise NotImplementedError
