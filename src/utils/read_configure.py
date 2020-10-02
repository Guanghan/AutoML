"""
load configuration file in yaml, and convert to Config class object
"""
import yaml


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
                _dict2config(self, raw_dict)


def _dict2config(config_dst: Config, dict_src: dict):
    """ Convert dictionary to config.

    Args:
        config_dst: Config obj to save info to
        dict_src: dict obj to load info from
    """
    if isinstance(dict_src, dict):
        for key, value in dict_src.items():
            if isinstance(value, dict):
                sub_config = Config()
                _dict2config(sub_config, value)
                dict.__setitem__(config_dst, key, sub_config)
            else:
                config_dst[key] = dict_src[key]
