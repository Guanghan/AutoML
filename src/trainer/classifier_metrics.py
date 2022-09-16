"""
@author: Guanghan Ning
@file: classifier_metrics.py
@time: 10/16/20 12:19
@file_desc: Metric of classification tasks.
"""
from .base_metrics import MetricBase
from src.core.class_factory import ClassFactory, ClassType
import torch.nn.functional as F
import numpy as np


def accuracy(output, target, top_k=(1,)):
    """Calculate classification accuracy between output and target.

    :param output: output of classification network
    :type output: pytorch tensor
    :param target: ground truth from dataset
    :type target: pytorch tensor
    :param top_k: top k of metric, k is an integer
    :type top_k: tuple of integer
    :return: results of top k
    :rtype: list

    """
    max_k = max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@ClassFactory.register(ClassType.METRIC, alias='accuracy')
class Accuracy(MetricBase):
    """Calculate classification accuracy between output and target."""

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1,)):
        """Init Accuracy metric."""
        self.topk = topk
        self.sum = [0.] * len(topk)
        self.data_num = 0
        self.pfm = [0.] * len(topk)

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        if isinstance(output, tuple):
            output = output[0]
        res = accuracy(output, target, self.topk)
        n = output.size(0)
        self.data_num += n
        self.sum = [self.sum[index] + item.item() * n for index, item in enumerate(res)]
        self.pfm = [item / self.data_num for item in self.sum]
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = [0.] * len(self.topk)
        self.data_num = 0
        self.pfm = [0.] * len(self.topk)

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        if len(self.pfm) == 1:
            return self.pfm[0]
        return {'top{}_{}'.format(self.topk[idx], self.name): value for idx, value in enumerate(self.pfm)}


def get_intercept_rate(pos_max_confs, neg_max_confs, through_rate=0.99):
    threshs = np.arange(0.01, 1.0, 0.01)

    pos_censor_list = []
    neg_through_list = []
    for thresh in threshs:
        neg_through_rate = 1 - get_censor_rate(neg_max_confs, thresh) if neg_max_confs != [] else 1
        pos_censor_rate = get_censor_rate(pos_max_confs, thresh) if pos_max_confs != [] else 0

        #print("thresh: {},  [official test set] pos_samples censor_rate: {:.1f}%, neg_samples through_rate: {:.1f}%".format(thresh, 100*pos_censor_rate, 100*neg_through_rate))
        pos_censor_list.append(pos_censor_rate)
        neg_through_list.append(abs(neg_through_rate - through_rate))

    index = np.argmin(neg_through_list, axis=0)
    #assert(neg_through_list[index] < 0.01)
    print("threshold: {}, censor_rate: {}, through_rate: {}".format(threshs[index], 100*pos_censor_list[index], 99-100*neg_through_list[index]))
    return 100*pos_censor_list[index]#, threshs[index]


def get_censor_rate(max_confs, thresh):
    censored = [max_conf for max_conf in max_confs if max_conf >= thresh]
    censor_rate = len(censored)*1.0 / len(max_confs)
    return censor_rate


def get_confs(output, cur_num):
    num_val = 1460 + 19327
    #print("confs: {}".format(output.cpu().detach().numpy()))
    if cur_num < 19327:
        pos_confs = []
        neg_confs = output.cpu().detach().numpy()[:, 1]
    else:
        pos_confs = output.cpu().detach().numpy()[:, 1]
        neg_confs = []
    return pos_confs, neg_confs


@ClassFactory.register(ClassType.METRIC, alias='intercept_rate')
class InterceptRate(MetricBase):
    """Calculate the interception rate of positive samples when the through rate of negative samples is 99%."""

    __metric_name__ = 'intercept_rate'

    def __init__(self, topk=(1,)):
        """Init Accuracy metric."""
        self.max_pos_confs = []
        self.max_neg_confs = []
        self.data_num = 0
        self.pfm = [0.]

    def __call__(self, output, target, *args, **kwargs):
        """Calculate interception rate.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: interception rate
        """
        #print("output.shape = {}".format(output.shape))
        if isinstance(output, tuple):
            output = output[0]

        output = F.softmax(output, dim =1)
        n = output.size(0)
        self.data_num += n
        #print("cur_num: {}".format(self.data_num))

        pos_confs, neg_confs = get_confs(output, self.data_num)
        #print("pos_confs = {}".format(pos_confs))
        #print("neg_confs = {}".format(neg_confs))
        #res = get_intercept_rate(pos_confs, neg_confs, 0.99)

        self.max_pos_confs.extend(pos_confs)
        self.max_neg_confs.extend(neg_confs)

        if self.data_num > 19327:
            self.pfm = [get_intercept_rate(self.max_pos_confs, self.max_neg_confs, 0.99)]
        #return res
        return

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.max_pos_confs = []
        self.neg_pos_confs = []
        self.data_num = 0
        self.pfm = [0.]

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        if len(self.pfm) == 1:
            return self.pfm[0]
        return {'{}'.format(self.name): value for idx, value in enumerate(self.pfm)}
