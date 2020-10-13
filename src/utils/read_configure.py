"""
load configuration file in yaml, and convert to Config class object
"""
import yaml
import copy
from inspect import isclass

import glog as log
from src.core.class_factory import ClassFactory, ClassType

class Config(dict):
    """
    Config class inherit from dict.
    It can parse arguments from a yaml file.
    """
    def __init__(self, yaml_path: str = None):
        """ A Config class inherited from dict

        Args:
            yaml_path: the path to the yaml configuration file
        """
        #super(Config, self).__init__()  # if python2
        super().__init__()
        if yaml_path is not None:
            assert (yaml_path.endswith('.yaml') or yaml_path.endswith('.yml'))
            with open(yaml_path) as file:
                # yaml load example: http://zetcode.com/python/yaml/
                raw_dict = yaml.load(file, Loader=yaml.FullLoader)
                dict2config(self, raw_dict)


def dict2config(config_dst: Config, dict_src: dict, is_clear = False):
    """ Convert dictionary to config.

    Args:
        config_dst: Config obj to save info to
        dict_src: dict obj to load info from
    """
    if not dict_src:
        raise ValueError("dict_src should be a dict.")

    # clear all attrs
    if is_clear:
        for attr_name in dir(config_dst):
            if attr_name.startswith('_'):
                continue
            delattr(config_dst, attr_name)

    # add attributes from dict_src to config_dst
    if isinstance(dict_src, dict):
        for key, value in dict_src.items():
            if isinstance(value, dict):
                sub_config = Config()
                dict2config(sub_config, value)
                dict.__setitem__(config_dst, key, sub_config)
            else:
                config_dst[key] = dict_src[key]


def class2config(config_dst: Config, class_src: object):
    """Convert obj to config.

    Args:
        obj(dict or class):

    """
    if not config_dst:
        config = Config()
    if isinstance(class_src, dict):
        log.info('In class2config, the given class is actually a dictionary.')
        dict_src = class_src
        return Config(dict_src)
    if not isinstance(class_src, object):
        log.info('In class2config, the given class is not an object.')
        return class_src
    for attr_name in dir(class_src):
        if attr_name.startswith('_'):
            continue
        if not isinstance(class_src, dict) and hasattr(class_src, '_exclude_keys'):
            exclude_keys = [class_src._exclude_keys] \
                           if not isinstance(class_src._exclude_keys, list) \
                           else class_src._exclude_keys
            if attr_name in exclude_keys:
                continue
        attr_value = getattr(class_src, attr_name)
        if isinstance(attr_value, type):
            attr_value = class2config(attr_value)
        config[attr_name] = attr_value
    return copy.deepcopy(config)


def desc2config(config_dst: Config, desc_src: dict):
    """ Convert description (dictionary) to config.

    Args:
        config_dst: Config obj to save info to
        desc_src: description, a special dict obj to load info from
    """
    if not isinstance(desc_src, dict):
        raise TypeError("desc should be a dict, desc={}".format(desc_src))
    desc_copy = copy.deepcopy(desc_src)
    # find the Config object according to desc
    for key, value in desc_copy.items():
        # reference other config objects
        if not hasattr(config_dst, key):
            setattr(config_dst, key, value)
        else:
            # use key as type_name
            sub_config_cls = getattr(config_dst, key)
            # Get config object dynamically according to type
            if not isinstance(sub_config_cls, dict) and hasattr(
                    sub_config_cls, '_class_type') and value and value.get('type'):
                ref_cls = ClassFactory.get_cls(sub_config_cls._class_type, value.type)
                if hasattr(ref_cls, 'config') and ref_cls.config and not isclass(ref_cls.config):
                    sub_config_cls = type(ref_cls.config)
            if not isclass(sub_config_cls) or value is None:
                setattr(config_dst, key, value)
            else:
                if hasattr(sub_config_cls, '_update_all_attrs') and sub_config_cls._update_all_attrs:
                    dict2config(sub_config_cls, value, is_clear=True)
                else:
                    desc2config(sub_config_cls, value)
