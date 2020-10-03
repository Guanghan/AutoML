"""
@author: Guanghan Ning
@file: test_class_factory.py
@time: 10/2/20 1:18 上午
@file_desc:
"""


def test_class_type():
    from src.core.class_factory import ClassType
    print(ClassType.__dict__)
    assert ClassType.GENERAL == "general"


def test_register():
    from src.core.class_factory import ClassFactory, ClassType
    @ClassFactory.register(ClassType.DATASET)
    class COCO(object):
        def __init__(self):
            print("COCO dataset")
    print(ClassFactory.__registry__)
    assert 'COCO' in ClassFactory.__registry__['dataset'].keys()


def test_register_cls():
    from src.core.class_factory import ClassFactory, ClassType
    class COCO(object):
        def __init__(self):
            print("COCO dataset")
    ClassFactory.register_cls(COCO, ClassType.DATASET)
    print(ClassFactory.__registry__)
    assert 'COCO' in ClassFactory.__registry__['dataset'].keys()


def test_get_cls():
    # Register
    from src.core.class_factory import ClassFactory, ClassType
    @ClassFactory.register(ClassType.DATASET)
    class COCO(object):
        def __init__(self):
            print("COCO dataset")

    # config
    from src.utils.read_configure import Config
    config = Config("../configs/example.yaml")
    print(config)

    # attach config to ClassFactory
    ClassFactory.attach_config_to_factory(config)
    print(ClassFactory.__configs__)

    # get corresponding classname given an attribute
    cls = ClassFactory.get_cls('dataset')
    import inspect
    assert inspect.isclass(cls) and cls.__name__ == 'COCO'


def test_is_exists():
    from src.core.class_factory import ClassFactory, ClassType
    @ClassFactory.register(ClassType.DATASET)
    class COCO(object):
        def __init__(self):
            print("COCO dataset")
    assert(ClassFactory.is_exists('dataset'))
