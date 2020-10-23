"""
@author: Guanghan Ning
@file: base_metrics.py
@time: 10/14/20 5:12
@file_desc: Base class for metrics. All metric classes are implemented with this base class.
"""

from functools import partial
from inspect import isfunction
from copy import deepcopy

from src.trainer import base_metrics as metrics
from src.utils.utils_cfg import Config, class2config
from src.core.class_factory import ClassFactory, ClassType
from src.core.default_config import MetricsConfig


class MetricBase(object):
    """Provide base metrics class for all custom metric to implement."""

    __metric_name__ = None

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy. called in train and valid step.

        Arguments:
            output: output of classification network
            target: ground truth from dataset
        """
        raise NotImplementedError

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        raise NotImplementedError

    def summary(self):
        """Summary all cached records, called after valid."""
        raise NotImplementedError

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    @property
    def name(self):
        """Get metric name."""
        return self.__metric_name__ or self.__class__.__name__

    @property
    def result(self):
        """Call summary to get result and parse result to dict.

        Return:
             dict like: {'acc':{'name': 'acc', 'value': 0.9, 'reward_mode': 'MAX'}}
        """
        value = self.summary()
        if isinstance(value, dict):
            return value
        return {self.name: value}


class Metrics(object):
    """Metrics class of all metrics defined in cfg.

     Arguments:
        metric_cfg (dict or Config): metric part of config
    """

    config = MetricsConfig()

    def __init__(self, metric_cfg=None):
        """Init Metrics."""
        self.mdict = {}
        #metric_config = class2config(self.config) if not metric_cfg else deepcopy(metric_cfg)
        if not metric_cfg:
            metric_config = class2config(Config(), self.config)
        else:
            metric_config = deepcopy(metric_cfg)

        if not isinstance(metric_config, list):
            metric_config = [metric_config]
        for metric_item in metric_config:
            ClassFactory.get_cls(ClassType.METRIC, self.config.type)
            metric_name = metric_item.pop('type')
            metric_class = ClassFactory.get_cls(ClassType.METRIC, metric_name)
            if isfunction(metric_class):
                metric_class = partial(metric_class, **metric_item.get("params", {}))
            else:
                metric_class = metric_class(**metric_item.get("params", {}))
            self.mdict[metric_name] = metric_class
        #self.mdict = Config(self.mdict)

    def __call__(self, output=None, target=None, *args, **kwargs):
        """Calculate all supported metrics by using output and target.

        Arguments:
            output (torch Tensor): predicted output by networks
            target (torch Tensor): target label data

        Return:
            performance of metrics (list)
        """
        pfms = []
        for key in self.mdict:
            metric = self.mdict[key]
            pfms.append(metric(output, target, *args, **kwargs))
        return pfms

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        for val in self.mdict.values():
            val.reset()

    @property
    def results(self):
        """Return metrics results."""
        res = {}
        for name, metric in self.mdict.items():
            res.update(metric.result)
        return res

    @property
    def objectives(self):
        """Return objectives results."""
        return {name: self.mdict.get(name).objective for name in self.mdict}

    def __getattr__(self, key):
        """Get a metric by key name.

        Arguments:
            key (str): metric name
        """
        return self.mdict[key]


ClassFactory.register_from_package(metrics, ClassType.METRIC)

