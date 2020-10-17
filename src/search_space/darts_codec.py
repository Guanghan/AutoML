"""
@author: Guanghan Ning
@file: darts_codec.py
@time: 10/6/20 10:37
@file_desc:
"""

import copy
import numpy as np
from src.utils.read_configure import dict2config, Config
from src.core.class_factory import ClassType, ClassFactory
from src.search_space.base_codec import Codec
import glog as log


@ClassFactory.register(ClassType.CODEC)
class DartsCodec(Codec):
    """
    The Compress/decompress for Darts network description
    """
    def __init__(self, search_space=None, **kwargs):
        super().__init__(search_space, **kwargs)
        self.darts_cfg = copy.deepcopy(search_space)
        log.info("self.darts_cfg: {}".format(self.darts_cfg))
        self.super_net = Config()
        super_net_dict = {'normal': self.darts_cfg["super_network"]["normal"]["genotype"],
                          'reduce': self.darts_cfg["super_network"]["reduce"]["genotype"]}
        dict2config(self.super_net, super_net_dict)
        self.steps = self.darts_cfg["super_network"]["normal"]["steps"]

    def decode(self, arch_param):
        """ Decode the alphas params to network description

        Args:
            arch_param (list of list of float):

        Returns: network description

        """
        genotype = self.calc_genotype(arch_param)
        cfg_result = copy.deepcopy(self.darts_cfg)
        cfg_result.super_network.normal.genotype = genotype[0]
        cfg_result.super_network.reduce.genotype = genotype[1]
        cfg_result.super_network.search = False
        cfg_result.super_network.auxiliary = True
        cfg_result.super_network["aux_size"] = 8
        cfg_result.super_network["auxiliary_layer"] = 13
        cfg_result.super_network.network = ["PreOneStem", "normal", "normal", "normal", "normal",
                                            "normal", "normal", "reduce", "normal", "normal", "normal", "normal",
                                            "normal", "normal",
                                            "reduce", "normal", "normal", "normal", "normal", "normal", "normal"]
        return cfg_result

    def calc_genotype(self, arch_param):
        """ Parse genotype from architecture (alpha) parameters
            From softmax to crisp: picking ops in cell

        Args:
            arch_param (list of list of float):

        Returns: two lists in the form of [str, int, int]

        """
        def _parse(weights, genos):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                G = genos[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if G[x][k] != 'none'))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if G[j][k] != 'none':
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append([G[j][k_best], i + 2, j])
                start = end
                n += 1
            return gene

        normal_param = np.array(self.darts_cfg["super_network"]["normal"]["genotype"])
        reduce_param = np.array(self.darts_cfg["super_network"]["reduce"]["genotype"])
        geno_normal = _parse(arch_param[0], normal_param[:, 0])
        geno_reduce = _parse(arch_param[1], reduce_param[:, 0])
        return geno_normal, geno_reduce
