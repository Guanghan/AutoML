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
    assert False


def test_get_cls():
    assert False


def test_is_exists():
    assert False
