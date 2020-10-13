"""
@author: Guanghan Ning
@file: search_space.py
@time: 10/9/20 3:30 下午
@file_desc:
"""
from src.core.class_factory import ClassFactory, ClassType
from src.utils.read_configure import class2config

class SearchSpaceConfig(object):
    """Default Search Space config for Pipeline."""

    _type_name = ClassType.SEARCH_SPACE
    type = None


@ClassFactory.register(ClassType.SEARCH_SPACE)
class SearchSpace(object):
    """Used for coarse search space. search space is the config from yaml."""

    config = SearchSpaceConfig()

    def __new__(cls, *args, **kwargs):
        """Create a new SearchSpace."""
        t_cls = ClassFactory.get_cls(ClassType.SEARCH_SPACE)
        return super(SearchSpace, cls).__new__(t_cls)

    @property
    def search_space(self):
        """Get hyper parameters."""
        return class2config(self.config)
