"""
@author: Guanghan Ning
@file: utils_dict.py
@time: 10/14/20 8:44 下午
@file_desc: Utilities related to dictionary
"""

from copy import deepcopy


def update_dict(src, dst, exclude= ['loss', 'metric', 'lr_scheduler', 'optim', 'model_desc', 'transforms']):
    """Use src dictionary update dst dictionary.

    Arguments:
        src (dict): Source dictionary.
        dst (dict): Dest dictionary.

    Return:
        Updated dictionary.
    """
    exclude_keys = exclude or []
    for key in src.keys():
        if key in dst.keys() and key not in exclude_keys:
            if isinstance(src[key], dict):
                dst[key] = update_dict(src[key], dst[key], exclude)
            else:
                dst[key] = src[key]
        else:
            dst[key] = src[key]
    return deepcopy(dst)
