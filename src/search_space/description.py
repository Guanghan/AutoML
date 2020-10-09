"""
@author: Guanghan Ning
@file: description.py
@time: 10/7/20 12:27 上午
@file_desc: Description of networks.
"""

"""Defined NetworkDesc."""
import hashlib
import logging
import json
from copy import deepcopy
from src.core.class_factory import ClassType, ClassFactory, NetworkType
from src.utils.read_configure import Config


class NetworkDesc(object):
    """Network Description."""

    def __init__(self, desc):
        """Init NetworkDesc."""
        self._desc = Config(deepcopy(desc))
        self._model_type = None
        self._model_name = None

    def to_model(self):
        """Transform a NetworkDesc to a special model."""
        if 'modules' not in self._desc:
            logging.debug("network=%s does not have key modules. desc={}".format(self._desc))
            return None

        networks = []
        module_types = self._desc.get('modules')
        if self._desc.get('type') != 'Network':
            for module_type in module_types:
                network = self.type_to_network(module_type)
                networks.append(network)
            if len(networks) == 1:
                return networks[0]
            else:
                import torch.nn as nn
                return nn.Sequential(*networks)

    def type_to_network(self, module_type):
        """Create network by module type."""
        module_desc = deepcopy(self._desc.get(module_type))
        if 'name' not in module_desc:
            raise KeyError('module descript does not have key {name}')

        module_name = module_desc.get('name')
        print("module_type: {}".format(module_type))
        if self._model_name is None:
            self._model_name = module_name
        if self._model_type is None:
            self._model_type = module_type

        network_cls = ClassFactory.get_cls(module_type, module_name)
        if network_cls is None:
            raise Exception("Network type error, module name: {}, module_type: {}".format(module_type, module_name))

        network = network_cls(module_desc)
        return network

    @property
    def md5(self):
        """MD5 value of network description."""
        return self.get_md5(self._desc)

    @classmethod
    def get_md5(cls, desc):
        """Get desc's short md5 code.

        Args:
            desc (str): network description.

        Return (str): short MD5 code.

        """
        _desc = deepcopy(desc)
        keys = ["modules"] + _desc["modules"]
        _desc = {key: _desc[key] for key in keys}
        code = hashlib.md5(json.dumps(_desc, sort_keys=True).encode('utf-8')).hexdigest()
        return code[:8]

    @property
    def model_type(self):
        """Return model type."""
        return self._model_type

    @property
    def model_name(self):
        """Return model name."""
        return self._model_name
