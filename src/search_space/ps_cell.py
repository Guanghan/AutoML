"""
@author: Guanghan Ning
@file: ps_cell.py
@time: 10/15/20 11:39 下午
@file_desc: parameter-sharing (ps) cells.
"""
import torch
import torch.nn as nn
from src.utils.read_configure import Config
from src.search_space.base_network import Network

from src.core.class_factory import NetworkType, ClassFactory
from src.search_space.common_ops import FactorizedReduce, ReluConvBn, Identity, drop_path


@ClassFactory.register(NetworkType.BLOCK)
class MixedOp(Network):
    """Mix operations between two nodes.

    :param desc: description of MixedOp
    :type desc: Config
    """

    def __init__(self, desc):
        """Init MixedOp."""
        super(MixedOp, self).__init__()
        C = desc["C"]
        stride = desc["stride"]
        ops_cands = desc["ops_cands"]
        if not isinstance(ops_cands, list):
            op_desc = {'C': C, 'stride': stride, 'affine': True}
            class_op = ClassFactory.get_cls(NetworkType.BLOCK, ops_cands)
            self._ops = class_op(Config(op_desc))
        else:
            self._ops = nn.ModuleList()
            for primitive in ops_cands:
                op_desc = {'C': C, 'stride': stride, 'affine': False}
                class_op = ClassFactory.get_cls(NetworkType.BLOCK, primitive)
                op = class_op(op_desc)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)

    def forward(self, x, weights=None):
        """Forward function of MixedOp."""
        if weights is not None:
            return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)
        else:
            return self._ops(x)


@ClassFactory.register(NetworkType.BLOCK)
class Cell(Network):
    """Cell structure according to desc.

    :param desc: description of Cell
    :type desc: Config
    """

    def __init__(self, desc):
        """Init Cell."""
        super(Cell, self).__init__()
        genotype = desc["genotype"]
        steps = desc["steps"]
        C_prev_prev = desc["C_prev_prev"]
        C_prev = desc["C_prev"]
        C = desc["C"]
        concat = desc["concat"]
        self.reduction = desc["reduction"]
        reduction_prev = desc["reduction_prev"]
        affine = True
        if isinstance(genotype[0][0], list):
            affine = False
        pre0_desc = self._pre_desc(C_prev_prev, C, 1, 1, 0, affine)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(pre0_desc)
        else:
            self.preprocess0 = ReluConvBn(pre0_desc)
        pre1_desc = self._pre_desc(C_prev, C, 1, 1, 0, affine)
        self.preprocess1 = ReluConvBn(pre1_desc)
        self._steps = steps
        self.search = desc["search"]
        op_names, indices_out, indices_inp = zip(*genotype)
        self._compile(C, op_names, indices_out, indices_inp, concat, self.reduction)

    def _pre_desc(self, channel_in, channel_out, kernel_size, stride, padding, affine):
        pre_desc = dict()
        pre_desc["channel_in"] = channel_in
        pre_desc["channel_out"] = channel_out
        pre_desc["affine"] = affine
        pre_desc["kernel_size"] = kernel_size
        pre_desc["stride"] = stride
        pre_desc["padding"] = padding
        return pre_desc

    def _compile(self, C, op_names, indices_out, indices_inp, concat, reduction):
        """Compile the cell.

        :param C: channels of this cell
        :type C: int
        :param op_names: list of all the operations in description
        :type op_names: list of str
        :param indices_out: list of all output nodes
        :type indices_out: list of int
        :param indices_inp: list of all input nodes link to output node
        :type indices_inp: list of int
        :param concat: cell concat list of output node
        :type concat: list of int
        :param reduction: whether to reduce
        :type reduction: bool
        """
        self._concat = concat
        self._multiplier = len(concat)
        self._ops = nn.ModuleList()
        self.out_inp_list = []
        temp_list = []
        idx_cmp = 2
        for i in range(len(op_names)):
            if indices_out[i] == idx_cmp:
                temp_list.append(indices_inp[i])
            elif indices_out[i] > idx_cmp:
                self.out_inp_list.append(temp_list.copy())
                temp_list = []
                idx_cmp += 1
                temp_list.append(indices_inp[i])
            else:
                raise Exception("input index should not less than idx_cmp")
            stride = 2 if reduction and indices_inp[i] < 2 else 1
            op = self.build_mixedop(C=C, stride=stride, ops_cands=op_names[i])
            self._ops.append(op)
        self.out_inp_list.append(temp_list.copy())
        if len(self.out_inp_list) != self._steps:
            raise Exception("out_inp_list length should equal to steps")

    def build_mixedop(self, **kwargs):
        """Build MixedOp.

        :param kwargs: arguments for MixedOp
        :type kwargs: dict
        :return: MixedOp Object
        :rtype: MixedOp
        """
        #mixedop_desc = Config(**kwargs)
        #return MixedOp(mixedop_desc)
        mixedop_desc = dict(**kwargs)
        return MixedOp(mixedop_desc)

    def forward(self, s0, s1, weights=None, drop_prob=0):
        """Forward function of Cell.

        :param s0: feature map of previous of previous cell
        :type s0: torch tensor
        :param s1: feature map of previous cell
        :type s1: torch tensor
        :param weights: weights of operations in cell
        :type weights: torch tensor, 2 dimension
        :return: cell output
        :rtype: torch tensor
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        drop = not self.search and drop_prob > 0.
        states = [s0, s1]
        idx = 0
        for i in range(self._steps):
            hlist = []
            for j, inp in enumerate(self.out_inp_list[i]):
                op = self._ops[idx + j]
                if weights is None:
                    h = op(states[inp])
                else:
                    h = op(states[inp], weights[idx + j])
                if drop and not isinstance(op._ops.block, Identity):
                    h = drop_path(h, drop_prob)
                hlist.append(h)
            s = sum(hlist)
            states.append(s)
            idx += len(self.out_inp_list[i])
        return torch.cat([states[i] for i in self._concat], dim=1)

