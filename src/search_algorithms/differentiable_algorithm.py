"""
@author: Guanghan Ning
@file: differentiable_algorithm.py
@time: 10/8/20 11:33 下午
@file_desc: Differentiable gradient method for neural architecture search (first proposed in DARTS)
"""

from src.core.class_factory import ClassType, ClassFactory
from src.search_algorithms.base_algorithm import SearchAlgorithm

@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DifferentialAlgorithm(SearchAlgorithm):
    """Differential algorithm."""
