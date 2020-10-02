"""
@author: Guanghan Ning
@file: class_factory.py
@time: 10/1/20 11:16 下午
@file_desc: Register classes so that a class can be instantiated via config file;
            no need to know explicitly which one.
"""

class ClassType(object):
    """Const class: saved defined class type."""
    # general: logger, GPU, work time, etc.
    GENERAL = 'general'
    # NAS and HPO
    SEARCH_ALGORITHM = 'search_algorithm'
    SEARCH_SPACE = 'search_space'
    # trainer
    TRAINER = 'trainer'
    METRIC = 'trainer.metric'
    OPTIM = 'trainer.optim'
    LR_SCHEDULER = 'trainer.lr_scheduler'
    LOSS = 'trainer.loss'
    # data
    DATASET = 'dataset'
    TRANSFORM = 'dataset.transforms'
    # evaluator
    EVALUATOR = 'evaluator'


class ClassFactory(object):
    """A Factory Class to manage all classes that need to register with config."""

    __registry__ = {}
    __configs__ = None

    @classmethod
    def register(cls, type_name=ClassType.GENERAL, alias=None):
        """ Register class with decorator, e.g.,
            @ClassFactory.register(ClassType.TRANSFORM)
            class Sharpness(object):

        Args:
            type_name (str): the name of the class type
            alias (str): alias name of the class

        Returns:
            wrapper(func):
        """
        def wrapper(t_cls):
            t_cls_name = alias if alias is not None else t_cls.__name__
            if type_name not in cls.__registry__:
                cls.__registry__[type_name] = {t_cls_name: t_cls}
            else:
                if t_cls_name in cls.__registry__:
                    raise ValueError("Cannot register duplicate class ({})".format(t_cls_name))
                cls.__registry__[type_name].update({t_cls_name: t_cls})
            return t_cls

        return wrapper

    @classmethod
    def register_cls(cls, t_cls, type_name=ClassType.GENERAL, alias=None):
        """ call this method to mannually register a class

        Args:
            t_cls (obj):  the class to register
            type_name (str): the name of the class type
            alias (str): alias name of the class

        Returns: t_cls (obj)

        """
        t_cls_name = alias if alias is not None else t_cls.__name__
        if type_name not in cls.__registry__:
            cls.__registry__[type_name] = {t_cls_name: t_cls}
        else:
            if t_cls_name in cls.__registry__:
                raise ValueError(
                    "Cannot register duplicate class ({})".format(t_cls_name))
            cls.__registry__[type_name].update({t_cls_name: t_cls})
        return t_cls

    @classmethod
    def get_cls(cls, type_name, t_cls_name=None):
        """ Get the class based on config

        Args:
            type_name: type name of class registry
            t_cls_name: (alternatively) class name can be explicitly given

        Returns: t_cls

        """
        if not cls.is_exists(type_name, t_cls_name):
            raise ValueError("can't find class type {} class name {} in class registry".format(type_name, t_cls_name))
        # create instance without configs
        if t_cls_name is None:
            t_cls_type = cls.__configs__
            for _type_name in type_name.split('.'):
                t_cls_type = t_cls_type.get(_type_name)
            t_cls_name = t_cls_type.get('type')
        if t_cls_name is None:
            raise ValueError("can't find class: {} with class type: {} in registry".format(t_cls_name, type_name))
        t_cls = cls.__registry__.get(type_name).get(t_cls_name)
        return t_cls

    @classmethod
    def is_exists(cls, type_name, cls_name=None):
        """ Check whether the class exists in the given type registry

        Args:
            type_name: type name of class registry
            t_cls_name: (alternatively) class name can be explicitly given

        Returns: True/False

        """
        if cls_name is None:
            return type_name in cls.__registry__
        return type_name in cls.__registry__ and cls_name in cls.__registry__.get(type_name)
