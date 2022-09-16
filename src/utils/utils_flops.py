"""
@author: Guanghan Ning
@file: utils_flops.py
@time: 10/22/20 8:53 下午
@file_desc: PyTorch model counter of FLOPS and parameters.
"""

from copy import deepcopy


def add_new_hooks(custom_hooks):
    """Add new register hooks to custom hooks."""
    import torch.nn as nn
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_softmax
    add_register_hooks = {
        nn.PReLU: register_hooks[nn.ReLU],
        nn.ELU: register_hooks[nn.ReLU],
        nn.Softmax: count_softmax
    }

    for k, v in add_register_hooks.items():
        if k not in register_hooks and k not in custom_hooks:
            custom_hooks[k] = v
    return custom_hooks


def calc_model_flops_params(model, input, custom_hooks=None, verbose=False):
    """Pytorch model flops and parameters calculation.

    :param model: pytorch model
    :type model: torch.nn.Module
    :param input: pytorch input tensor
    :type input: torch.Tensor
    :param custom_hooks: hooks defined by outside customer
    :type custom_hooks: dict or None
    :param verbose: whether to print op type which not in collection
    :type verbose: bool, default True
    :return: flops and params
    :rtype: float, float
    """
    from thop import profile
    _model = deepcopy(model)
    if custom_hooks is None:
        custom_hooks = {}
    custom_hooks = add_new_hooks(custom_hooks)
    inputs = (input,)
    flops, params = profile(_model, inputs, custom_hooks, verbose)
    del _model
    return flops, params

