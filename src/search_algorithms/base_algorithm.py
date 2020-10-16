"""
@author: Guanghan Ning
@file: base_algorithm.py
@time: 10/8/20 11:23 下午
@file_desc: Base search algorithm class. Search algorithms (e.g., darts) inherit from this algorithm.
"""
import glog as log
log.setLevel("INFO")

from src.core.class_factory import ClassType, ClassFactory
from src.core.base_task import Task
from src.utils.read_configure import desc2config
from src.search_space.base_codec import Codec


class SearchAlgorithm(Task):
    """SearchAlgorithm the base class for user defined search algorithms.

    Args:
        search_space (SearchSpace): User defined `search_space`, default is None.

    Attributes:
        self.config
        self.codec
        self.search_space

    Properties:
        self.is_completed
    """

    config = None

    def __new__(cls, *args, **kwargs):
        """Create search algorithm instance by ClassFactory."""
        if cls.__name__ != 'SearchAlgorithm':
            return super().__new__(cls)
        if kwargs.get('type'):
            t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM, kwargs.pop('type'))
        else:
            t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM)

        return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        """Init SearchAlgorithm.

        Args:
            search_space (SearchSpace): initialize the algorithm with a search space
        """
        super(SearchAlgorithm, self).__init__()

        # modify config by kwargs in local scope
        if self.config and kwargs:
            self.config = self.config() #TODO
            desc2config(self.config, kwargs)
        log.info("Search algorithm config check.")

        self.search_space = search_space

        if hasattr(self.config, 'codec'):
            self.codec = Codec(search_space, type=self.config.codec)
        else:
            self.codec = None

    def search(self):
        """Search function."""
        raise NotImplementedError

    @property
    def is_completed(self):
        """If the search is finished."""
        raise NotImplementedError
