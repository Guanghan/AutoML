"""
@author: Guanghan Ning
@file: test_default_config.py
@time: 10/15/20 6:23 下午
@file_desc:
"""


def test_optim_config():
    from src.core.default_config import OptimConfig
    assert OptimConfig.params == {'lr': 0.1}
