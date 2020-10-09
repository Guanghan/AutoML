"""
@author: Guanghan Ning
@file: base_codec.py
@time: 10/5/20 9:33 下午
@file_desc: Define basic codec
"""

from src.core.class_factory import ClassType, ClassFactory

class Codec(object):
    """
    The base class for compress/decompress algorithms
    """
    def __new__(cls, *args, **kwargs):
        """Create search algorithm instance by ClassFactory."""
        if cls.__name__ != 'Codec':
            # if base codec, create instance
            return super().__new__(cls)
        else:
            # if inherited codec, create corresponding instance based on registered class in class factory
            t_cls = ClassFactory.get_cls(ClassType.CODEC)
            return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        self.search_space = search_space

    def encode(self, desc):
        raise NotImplementedError

    def decode(self, code):
        raise NotImplementedError
